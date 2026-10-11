target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
define internal i32 @global_id() #1 { entry:
%lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() %group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %base = mul i32 %group, %width %id = add i32 %base, %lane ret i32 %id }
@grid.count = internal addrspace(1) global i32 0, align 4
@grid.phase = internal addrspace(1) global i32 0, align 4
define internal void @grid_barrier(i32 %threads) #1 { entry:
; Every writer releases its own global-memory operations before the block arrives.
fence syncscope("device") release
%phase = load atomic i32, ptr addrspace(1) @grid.phase syncscope("device") monotonic, align 4
call void @llvm.nvvm.barrier0() %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%lane = and i32 %tid, 31 %spinner = icmp eq i32 %lane, 0
%leader = icmp eq i32 %tid, 0 br i1 %leader, label %arrive, label %check arrive:
%width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %groups = udiv i32 %threads, %width
fence syncscope("device") release
%prior = atomicrmw add ptr addrspace(1) @grid.count, i32 1 syncscope("device") monotonic %limit = sub i32 %groups, 1
%last = icmp eq i32 %prior, %limit br i1 %last, label %release, label %wait release:
fence syncscope("device") acquire
store atomic i32 0, ptr addrspace(1) @grid.count syncscope("device") monotonic, align 4 %next = xor i32 %phase, 1
fence syncscope("device") release
store atomic i32 %next, ptr addrspace(1) @grid.phase syncscope("device") monotonic, align 4 br label %wait check:
br i1 %spinner, label %wait, label %waited wait:
%seen = load atomic i32, ptr addrspace(1) @grid.phase syncscope("device") monotonic, align 4 %ready = icmp ne i32 %seen, %phase
br i1 %ready, label %acquired, label %wait acquired:
fence syncscope("device") acquire br label %waited waited:
call void @llvm.nvvm.barrier0()
fence syncscope("device") acquire
ret void }
; NUMERIC BEGIN
declare float @llvm.sqrt.f32(float) declare float @llvm.fabs.f32(float) declare float @llvm.floor.f32(float) declare float @llvm.roundeven.f32(float)
define internal float @recipe.decode(i8 %value) #3 {
entry:
%wide = zext i8 %value to i32
%magnitude = and i32 %wide, 127
%sign = and i32 %wide, 128
%float.sign = shl i32 %sign, 24
%exponent = lshr i32 %magnitude, 3
%mantissa = and i32 %wide, 7
%float.mantissa = shl i32 %mantissa, 20
%zero.exponent = icmp eq i32 %exponent, 0
br i1 %zero.exponent, label %subnormal, label %normal
subnormal:
%mantissa.float = uitofp i32 %mantissa to float
%scaled = fmul float %mantissa.float, 0x3F60000000000000
%sub.bits = bitcast float %scaled to i32
%sub.signed = or i32 %sub.bits, %float.sign
%sub.value = bitcast i32 %sub.signed to float
ret float %sub.value
normal:
%biased = add i32 %exponent, 120
%exponent.bits = shl i32 %biased, 23
%normal.base = or i32 %float.sign, %exponent.bits
%normal.bits = or i32 %normal.base, %float.mantissa
%special = icmp eq i32 %magnitude, 127
%special.bits = or i32 %float.sign, 2143289344
%bits = select i1 %special, i32 %special.bits, i32 %normal.bits
%result = bitcast i32 %bits to float
ret float %result
}
define internal i8 @recipe.fp8.encode(double %value) #1 {
entry:
%bits = bitcast double %value to i64
%sign.wide = lshr i64 %bits, 56
%sign = and i64 %sign.wide, 128
%absolute = and i64 %bits, 9223372036854775807
%exponent.field = lshr i64 %absolute, 52
%mantissa = and i64 %absolute, 4503599627370495
%nan = icmp ugt i64 %absolute, 9218868437227405312
br i1 %nan, label %quiet, label %finite.check
quiet:
%quiet.bits = or i64 %sign, 127
br label %pack
finite.check:
%unbiased = sub i64 %exponent.field, 1023
%target = add i64 %unbiased, 7
%too.large = icmp sge i64 %target, 16
br i1 %too.large, label %over, label %range
over:
%over.bits = or i64 %sign, 126
br label %pack
range:
%normal = icmp sge i64 %target, 1
br i1 %normal, label %round.normal, label %round.small
round.normal:
%keep = lshr i64 %mantissa, 49
%rest = and i64 %mantissa, 562949953421311
%rest.above = icmp ugt i64 %rest, 281474976710656
%rest.tie = icmp eq i64 %rest, 281474976710656
%keep.odd = trunc i64 %keep to i1
%tie.up = and i1 %rest.tie, %keep.odd
%up = or i1 %rest.above, %tie.up
%up.wide = zext i1 %up to i64
%rounded = add i64 %keep, %up.wide
%exponent.bits = shl i64 %target, 3
%normal.bits.raw = add i64 %exponent.bits, %rounded
%normal.over = icmp sge i64 %normal.bits.raw, 127
%normal.bits.kept = select i1 %normal.over, i64 126, i64 %normal.bits.raw
%normal.bits = or i64 %sign, %normal.bits.kept
br label %pack
round.small:
%full = or i64 %mantissa, 4503599627370496
%below = sub i64 1, %target
%shift.raw = add i64 %below, 49
%shift.far = icmp ugt i64 %shift.raw, 62
%shift = select i1 %shift.far, i64 62, i64 %shift.raw
%small.keep = lshr i64 %full, %shift
%shift.less = sub i64 %shift, 1
%small.half = shl i64 1, %shift.less
%small.mask = sub i64 %small.half, 1
%small.rest.mask = or i64 %small.mask, %small.half
%small.rest = and i64 %full, %small.rest.mask
%small.above = icmp ugt i64 %small.rest, %small.half
%small.tie = icmp eq i64 %small.rest, %small.half
%small.odd = trunc i64 %small.keep to i1
%small.tie.up = and i1 %small.tie, %small.odd
%small.up = or i1 %small.above, %small.tie.up
%small.up.wide = zext i1 %small.up to i64
%small.rounded = add i64 %small.keep, %small.up.wide
%small.bits = or i64 %sign, %small.rounded
br label %pack
pack:
%packed.wide = phi i64 [ %quiet.bits, %quiet ], [ %over.bits, %over ], [ %normal.bits, %round.normal ], [ %small.bits, %round.small ]
%packed = trunc i64 %packed.wide to i8
%result = add i8 %packed, 0
ret i8 %result
}

define internal i8 @recipe.encode(float %value) #3 { entry:
%wide = fpext float %value to double
%result = call i8 @recipe.fp8.encode(double %wide)
ret i8 %result
}


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
define internal double @recipe.math.log.wide(double %x) #1 {
entry:
%unordered = fcmp uno double %x, %x
br i1 %unordered, label %quiet, label %sign.check
quiet:
ret double %x
sign.check:
%negative = fcmp olt double %x, 0.0
br i1 %negative, label %invalid, label %zero.check
invalid:
ret double 0x7FF8000000000000
zero.check:
%zero = fcmp oeq double %x, 0.0
br i1 %zero, label %pole, label %infinite.check
pole:
ret double 0xFFF0000000000000
infinite.check:
%infinite = fcmp oeq double %x, 0x7FF0000000000000
br i1 %infinite, label %unbounded, label %normalize
unbounded:
ret double 0x7FF0000000000000
normalize:
%raw = bitcast double %x to i64
%subnormal = icmp ult i64 %raw, 4503599627370496
%boosted = fmul double %x, 0x4350000000000000
%source = select i1 %subnormal, double %boosted, double %x
%correction = select i1 %subnormal, i32 -54, i32 0
%bits = bitcast double %source to i64
%field = lshr i64 %bits, 52
%narrow = trunc i64 %field to i32
%unbiased = sub i32 %narrow, 1023
%exponent = add i32 %unbiased, %correction
%fraction = and i64 %bits, 4503599627370495
%unit = or i64 %fraction, 4607182418800017408
%m = bitcast i64 %unit to double
%large = fcmp ogt double %m, 0x3FF6A09E667F3BCD
%halved = fmul double %m, 0.5
%mantissa = select i1 %large, double %halved, double %m
%bump = select i1 %large, i32 1, i32 0
%e = add i32 %exponent, %bump
%ef = sitofp i32 %e to double
%numerator = fsub double %mantissa, 1.0
%denominator = fadd double %mantissa, 1.0
%s = fdiv double %numerator, %denominator
%z = fmul double %s, %s
%log.poly.0 = call double @llvm.fma.f64(double 0x3FA642C8590B2164, double %z, double 0x3FA8618618618618)
%log.poly.1 = call double @llvm.fma.f64(double %log.poly.0, double %z, double 0x3FAAF286BCA1AF28)
%log.poly.2 = call double @llvm.fma.f64(double %log.poly.1, double %z, double 0x3FAE1E1E1E1E1E1E)
%log.poly.3 = call double @llvm.fma.f64(double %log.poly.2, double %z, double 0x3FB1111111111111)
%log.poly.4 = call double @llvm.fma.f64(double %log.poly.3, double %z, double 0x3FB3B13B13B13B14)
%log.poly.5 = call double @llvm.fma.f64(double %log.poly.4, double %z, double 0x3FB745D1745D1746)
%log.poly.6 = call double @llvm.fma.f64(double %log.poly.5, double %z, double 0x3FBC71C71C71C71C)
%log.poly.7 = call double @llvm.fma.f64(double %log.poly.6, double %z, double 0x3FC2492492492492)
%log.poly.8 = call double @llvm.fma.f64(double %log.poly.7, double %z, double 0x3FC999999999999A)
%log.poly.9 = call double @llvm.fma.f64(double %log.poly.8, double %z, double 0x3FD5555555555555)
%log.poly.10 = call double @llvm.fma.f64(double %log.poly.9, double %z, double 0x3FF0000000000000)
%series = fmul double %s, %log.poly.10
%twice = fmul double %series, 2.0
%upper = fmul double %ef, 0x3FE62E42FEE00000
%lower = fmul double %ef, 0x3DEA39EF35793C76
%tail = fadd double %twice, %lower
%result = fadd double %upper, %tail
ret double %result
}
define internal double @recipe.math.trig(double %x, i32 %offset) #1 {
entry:
%unordered = fcmp uno double %x, %x
%absolute = call double @llvm.fabs.f64(double %x)
%infinite = fcmp oeq double %absolute, 0x7FF0000000000000
%invalid = or i1 %unordered, %infinite
br i1 %invalid, label %undefined, label %reduce
undefined:
ret double 0x7FF8000000000000
reduce:
%scaled = fmul double %x, 0x3FE45F306DC9C883
%shifted = fadd double %scaled, 0.5
%kf = call double @llvm.floor.f64(double %shifted)
%first = fmul double %kf, 0x3FF921FB54400000
%partial = fsub double %x, %first
%second = fmul double %kf, 0x3DD0B4611A600000
%closer = fsub double %partial, %second
%third = fmul double %kf, 0x3BA3198A2E000000
%r = fsub double %closer, %third
%quarter = fmul double %kf, 0.25
%whole = call double @llvm.floor.f64(double %quarter)
%rounds = fmul double %whole, 4.0
%residue = fsub double %kf, %rounds
%k = fptosi double %residue to i32
%shift = add i32 %k, %offset
%quadrant = and i32 %shift, 3
%z = fmul double %r, %r
%sin.poly.0 = call double @llvm.fma.f64(double 0xBC62F49B46814157, double %z, double 0x3CE952C77030AD4A)
%sin.poly.1 = call double @llvm.fma.f64(double %sin.poly.0, double %z, double 0xBD6AE7F3E733B81F)
%sin.poly.2 = call double @llvm.fma.f64(double %sin.poly.1, double %z, double 0x3DE6124613A86D09)
%sin.poly.3 = call double @llvm.fma.f64(double %sin.poly.2, double %z, double 0xBE5AE64567F544E4)
%sin.poly.4 = call double @llvm.fma.f64(double %sin.poly.3, double %z, double 0x3EC71DE3A556C734)
%sin.poly.5 = call double @llvm.fma.f64(double %sin.poly.4, double %z, double 0xBF2A01A01A01A01A)
%sin.poly.6 = call double @llvm.fma.f64(double %sin.poly.5, double %z, double 0x3F81111111111111)
%sin.poly.7 = call double @llvm.fma.f64(double %sin.poly.6, double %z, double 0xBFC5555555555555)
%sin.body = fmul double %z, %sin.poly.7
%sin.scaled = fmul double %r, %sin.body
%sin = fadd double %r, %sin.scaled
%cos.poly.0 = call double @llvm.fma.f64(double 0xBCA6827863B97D97, double %z, double 0x3D2AE7F3E733B81F)
%cos.poly.1 = call double @llvm.fma.f64(double %cos.poly.0, double %z, double 0xBDA93974A8C07C9D)
%cos.poly.2 = call double @llvm.fma.f64(double %cos.poly.1, double %z, double 0x3E21EED8EFF8D898)
%cos.poly.3 = call double @llvm.fma.f64(double %cos.poly.2, double %z, double 0xBE927E4FB7789F5C)
%cos.poly.4 = call double @llvm.fma.f64(double %cos.poly.3, double %z, double 0x3EFA01A01A01A01A)
%cos.poly.5 = call double @llvm.fma.f64(double %cos.poly.4, double %z, double 0xBF56C16C16C16C17)
%cos.poly.6 = call double @llvm.fma.f64(double %cos.poly.5, double %z, double 0x3FA5555555555555)
%cos.poly.7 = call double @llvm.fma.f64(double %cos.poly.6, double %z, double 0xBFE0000000000000)
%cos.body = fmul double %z, %cos.poly.7
%cos = fadd double 1.0, %cos.body
%parity = and i32 %quadrant, 1
%cosine = icmp eq i32 %parity, 1
%magnitude = select i1 %cosine, double %cos, double %sin
%upper = icmp uge i32 %quadrant, 2
%negated = fneg double %magnitude
%result = select i1 %upper, double %negated, double %magnitude
ret double %result
}
define internal double @recipe.math.sin.wide(double %x) #1 { entry: %result = call double @recipe.math.trig(double %x, i32 0) ret double %result }
define internal double @recipe.math.cos.wide(double %x) #1 { entry: %result = call double @recipe.math.trig(double %x, i32 1) ret double %result }
define internal float @recipe.math.exp(float %value) #1 { entry: %result = call float @recipe.math.exp.narrow(float %value) ret float %result }
define internal float @recipe.math.tanh(float %value) #1 { entry: %wide = fpext float %value to double %computed = call double @recipe.math.tanh.wide(double %wide) %result = fptrunc double %computed to float ret float %result }
define internal float @recipe.math.cos(float %value) #1 { entry: %wide = fpext float %value to double %computed = call double @recipe.math.cos.wide(double %wide) %result = fptrunc double %computed to float ret float %result }
define internal float @recipe.math.sin(float %value) #1 { entry: %wide = fpext float %value to double %computed = call double @recipe.math.sin.wide(double %wide) %result = fptrunc double %computed to float ret float %result }
define internal float @recipe.math.log(float %value) #1 { entry: %wide = fpext float %value to double %computed = call double @recipe.math.log.wide(double %wide) %result = fptrunc double %computed to float ret float %result }
define internal i8 @recipe.add(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %wide = fadd float %left.wide, %right.wide %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i8 @recipe.sub(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %wide = fsub float %left.wide, %right.wide %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i8 @recipe.mul(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %wide = fmul float %left.wide, %right.wide %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i8 @recipe.div(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %wide = fdiv float %left.wide, %right.wide %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i8 @recipe.madd(i8 %sum, i8 %left, i8 %right) #1 { entry: %sum.wide = call float @recipe.decode(i8 %sum) %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %wide = call float @llvm.fma.f32(float %left.wide, float %right.wide, float %sum.wide) %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal <RECIPE_REGISTER_M x i8> @recipe.madd.vector(<RECIPE_REGISTER_M x i8> %sum, <RECIPE_REGISTER_M x i8> %left, <RECIPE_REGISTER_M x i8> %right) #1 {
entry:
br label %loop
loop:
%p = phi i32 [ 0, %entry ], [ %p.next, %step ]
%result = phi <RECIPE_REGISTER_M x i8> [ poison, %entry ], [ %next, %step ]
%more = icmp ult i32 %p, RECIPE_REGISTER_M
br i1 %more, label %step, label %done
step:
%sum.value = extractelement <RECIPE_REGISTER_M x i8> %sum, i32 %p
%left.value = extractelement <RECIPE_REGISTER_M x i8> %left, i32 %p
%right.value = extractelement <RECIPE_REGISTER_M x i8> %right, i32 %p
%value = call i8 @recipe.madd(i8 %sum.value, i8 %left.value, i8 %right.value)
%next = insertelement <RECIPE_REGISTER_M x i8> %result, i8 %value, i32 %p
%p.next = add i32 %p, 1
br label %loop
done:
ret <RECIPE_REGISTER_M x i8> %result
}
define internal i8 @recipe.neg(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %negative = fneg float %wide %result = call i8 @recipe.encode(float %negative) ret i8 %result }
define internal i1 @recipe.oeq(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %result = fcmp oeq float %left.wide, %right.wide ret i1 %result }
define internal i1 @recipe.oge(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %result = fcmp oge float %left.wide, %right.wide ret i1 %result }
define internal i1 @recipe.ogt(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %result = fcmp ogt float %left.wide, %right.wide ret i1 %result }
define internal i1 @recipe.ole(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %result = fcmp ole float %left.wide, %right.wide ret i1 %result }
define internal i1 @recipe.olt(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %result = fcmp olt float %left.wide, %right.wide ret i1 %result }
define internal i1 @recipe.one(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %result = fcmp one float %left.wide, %right.wide ret i1 %result }
define internal i1 @recipe.ord(i8 %left, i8 %right) #1 { entry: %left.wide = call float @recipe.decode(i8 %left) %right.wide = call float @recipe.decode(i8 %right) %result = fcmp ord float %left.wide, %right.wide ret i1 %result }
define internal i8 @recipe.from.u1(i1 %value) #1 { entry: %wide = uitofp i1 %value to float %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i8 @recipe.from.u32(i32 %value) #1 { entry: %wide = uitofp i32 %value to float %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i8 @recipe.from.s32(i32 %value) #1 { entry: %wide = sitofp i32 %value to float %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i32 @recipe.to.u32(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %result = fptoui float %wide to i32 ret i32 %result }
define internal i32 @recipe.to.s32(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %result = fptosi float %wide to i32 ret i32 %result }
define internal i8 @recipe.from.f32(float %value) #1 { entry: %wide = fadd float %value, 0.0 %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal i8 @recipe.from.f16(half %value) #1 { entry: %wide = fpext half %value to float %result = call i8 @recipe.encode(float %wide) ret i8 %result }
define internal half @recipe.to.f16(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %result = fptrunc float %wide to half ret half %result }
define internal float @recipe.to.f32(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %result = fadd float %wide, 0.0 ret float %result }
define internal i8 @recipe.abs(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @llvm.fabs.f32(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.floor(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @llvm.floor.f32(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.roundeven(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @llvm.roundeven.f32(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.sqrt(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @llvm.sqrt.f32(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.exp(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @recipe.math.exp(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.tanh(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @recipe.math.tanh(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.cos(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @recipe.math.cos(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.sin(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @recipe.math.sin(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.log(i8 %value) #1 { entry: %wide = call float @recipe.decode(i8 %value) %computed = call float @recipe.math.log(float %wide) %result = call i8 @recipe.encode(float %computed) ret i8 %result }
define internal i8 @recipe.sigmoid(i8 %value) #1 { entry: %negative = call i8 @recipe.neg(i8 %value) %exponential = call i8 @recipe.exp(i8 %negative) %one = call i8 @recipe.from.u1(i1 true) %denominator = call i8 @recipe.add(i8 %exponential, i8 %one) %result = call i8 @recipe.div(i8 %one, i8 %denominator) ret i8 %result }
define internal void @recipe.set.format(i32 %exp, i32 %man) #1 { entry: ret void }
define internal float @recipe.state.add(float %left, float %right) #1 { entry: %result = fadd float %left, %right ret float %result }
define internal float @recipe.state.sub(float %left, float %right) #1 { entry: %result = fsub float %left, %right ret float %result }
define internal float @recipe.state.mul(float %left, float %right) #1 { entry: %result = fmul float %left, %right ret float %result }
define internal float @recipe.state.div(float %left, float %right) #1 { entry: %result = fdiv float %left, %right ret float %result }
define internal float @recipe.state.madd(float %sum, float %left, float %right) #1 { entry: %result = call float @llvm.fma.f32(float %left, float %right, float %sum) ret float %result }
declare <RECIPE_REGISTER_M x float> @llvm.fma.vRECIPE_REGISTER_Mf32(<RECIPE_REGISTER_M x float>, <RECIPE_REGISTER_M x float>, <RECIPE_REGISTER_M x float>)
define internal <RECIPE_REGISTER_M x float> @recipe.state.madd.vector(<RECIPE_REGISTER_M x float> %sum, <RECIPE_REGISTER_M x float> %left, <RECIPE_REGISTER_M x float> %right) #1 {
entry:
%result = call <RECIPE_REGISTER_M x float> @llvm.fma.vRECIPE_REGISTER_Mf32(<RECIPE_REGISTER_M x float> %left, <RECIPE_REGISTER_M x float> %right, <RECIPE_REGISTER_M x float> %sum)
ret <RECIPE_REGISTER_M x float> %result
}
define internal float @recipe.state.neg(float %value) #1 { entry: %result = fneg float %value ret float %result }
define internal i1 @recipe.state.oeq(float %left, float %right) #1 { entry: %result = fcmp oeq float %left, %right ret i1 %result }
define internal i1 @recipe.state.oge(float %left, float %right) #1 { entry: %result = fcmp oge float %left, %right ret i1 %result }
define internal i1 @recipe.state.ogt(float %left, float %right) #1 { entry: %result = fcmp ogt float %left, %right ret i1 %result }
define internal i1 @recipe.state.ole(float %left, float %right) #1 { entry: %result = fcmp ole float %left, %right ret i1 %result }
define internal i1 @recipe.state.olt(float %left, float %right) #1 { entry: %result = fcmp olt float %left, %right ret i1 %result }
define internal i1 @recipe.state.one(float %left, float %right) #1 { entry: %result = fcmp one float %left, %right ret i1 %result }
define internal i1 @recipe.state.ord(float %left, float %right) #1 { entry: %result = fcmp ord float %left, %right ret i1 %result }
define internal float @recipe.state.from.u1(i1 %value) #1 { entry: %result = uitofp i1 %value to float ret float %result }
define internal float @recipe.state.from.u32(i32 %value) #1 { entry: %result = uitofp i32 %value to float ret float %result }
define internal float @recipe.state.from.s32(i32 %value) #1 { entry: %result = sitofp i32 %value to float ret float %result }
define internal i32 @recipe.state.to.u32(float %value) #1 { entry: %result = fptoui float %value to i32 ret i32 %result }
define internal i32 @recipe.state.to.s32(float %value) #1 { entry: %result = fptosi float %value to i32 ret i32 %result }
define internal float @recipe.state.from.f32(float %value) #1 { entry: %wide = fadd float %value, 0.0 ret float %wide }
define internal float @recipe.state.from.f16(half %value) #1 { entry: %wide = fpext half %value to float ret float %wide }
define internal half @recipe.state.to.f16(float %value) #1 { entry: %wide = fadd float %value, 0.0 %result = fptrunc float %wide to half ret half %result }
define internal float @recipe.state.to.f32(float %value) #1 { entry: %wide = fadd float %value, 0.0 %result = fadd float %wide, 0.0 ret float %result }
define internal float @recipe.state.abs(float %value) #1 { entry: %result = call float @llvm.fabs.f32(float %value) ret float %result }
define internal float @recipe.state.floor(float %value) #1 { entry: %result = call float @llvm.floor.f32(float %value) ret float %result }
define internal float @recipe.state.roundeven(float %value) #1 { entry: %result = call float @llvm.roundeven.f32(float %value) ret float %result }
define internal float @recipe.state.sqrt(float %value) #1 { entry: %result = call float @llvm.sqrt.f32(float %value) ret float %result }
define internal float @recipe.state.exp(float %value) #1 { entry: %result = call float @recipe.math.exp(float %value) ret float %result }
define internal float @recipe.state.tanh(float %value) #1 { entry: %result = call float @recipe.math.tanh(float %value) ret float %result }
define internal float @recipe.state.cos(float %value) #1 { entry: %result = call float @recipe.math.cos(float %value) ret float %result }
define internal float @recipe.state.sin(float %value) #1 { entry: %result = call float @recipe.math.sin(float %value) ret float %result }
define internal float @recipe.state.log(float %value) #1 { entry: %result = call float @recipe.math.log(float %value) ret float %result }
define internal float @recipe.state.sigmoid(float %value) #1 { entry: %negative = call float @recipe.state.neg(float %value) %exponential = call float @recipe.state.exp(float %negative) %one = call float @recipe.state.from.u1(i1 true) %denominator = call float @recipe.state.add(float %exponential, float %one) %result = call float @recipe.state.div(float %one, float %denominator) ret float %result }
define internal float @recipe.state.from.model(i8 %value) #1 { entry: %result = call float @recipe.decode(i8 %value) ret float %result }
define internal i8 @recipe.model.from.state(float %value) #1 { entry: %result = call i8 @recipe.encode(float %value) ret i8 %result }
; NUMERIC END
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0() declare i64 @__ockl_steadyctr_u64()
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32)
declare i64 @llvm.nvvm.read.ptx.sreg.globaltimer()
define internal i32 @recipe.wavefront.width() #1 { entry: ret i32 32 }
define internal i1 @recipe.int8.dots() #1 { entry: ret i1 true }
define internal i64 @recipe.clock() #1 { entry: %now = call i64 @llvm.nvvm.read.ptx.sreg.globaltimer() ret i64 %now }
define internal float @recipe.wave.partner(float %value, i32 %index) #1 { entry: %lane = lshr i32 %index, 2 %bits = bitcast float %value to i32 %partner.bits = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %bits, i32 %lane, i32 31) %partner = bitcast i32 %partner.bits to float ret float %partner }
define internal float @recipe.wave.partner.f32(float %value, i32 %index) #1 { entry: %lane = lshr i32 %index, 2 %bits = bitcast float %value to i32 %partner.bits = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %bits, i32 %lane, i32 31) %partner = bitcast i32 %partner.bits to float ret float %partner }
define internal i32 @recipe.dot4.su(i32 %a, i32 %b) #1 { entry:
%a0.s = lshr i32 %a, 0
%a0 = and i32 %a0.s, 255
%b0.s = shl i32 %b, 24
%b0 = ashr i32 %b0.s, 24
%p0 = mul i32 %a0, %b0
%s0 = add i32 0, %p0
%a1.s = lshr i32 %a, 8
%a1 = and i32 %a1.s, 255
%b1.s = shl i32 %b, 16
%b1 = ashr i32 %b1.s, 24
%p1 = mul i32 %a1, %b1
%s1 = add i32 %s0, %p1
%a2.s = lshr i32 %a, 16
%a2 = and i32 %a2.s, 255
%b2.s = shl i32 %b, 8
%b2 = ashr i32 %b2.s, 24
%p2 = mul i32 %a2, %b2
%s2 = add i32 %s1, %p2
%a3.s = lshr i32 %a, 24
%a3 = and i32 %a3.s, 255
%b3.s = shl i32 %b, 0
%b3 = ashr i32 %b3.s, 24
%p3 = mul i32 %a3, %b3
%s3 = add i32 %s2, %p3
ret i32 %s3
}
define internal i32 @recipe.dot4.ss(i32 %a, i32 %b) #1 { entry:
%a0.s = shl i32 %a, 24
%a0 = ashr i32 %a0.s, 24
%b0.s = shl i32 %b, 24
%b0 = ashr i32 %b0.s, 24
%p0 = mul i32 %a0, %b0
%s0 = add i32 0, %p0
%a1.s = shl i32 %a, 16
%a1 = ashr i32 %a1.s, 24
%b1.s = shl i32 %b, 16
%b1 = ashr i32 %b1.s, 24
%p1 = mul i32 %a1, %b1
%s1 = add i32 %s0, %p1
%a2.s = shl i32 %a, 8
%a2 = ashr i32 %a2.s, 24
%b2.s = shl i32 %b, 8
%b2 = ashr i32 %b2.s, 24
%p2 = mul i32 %a2, %b2
%s2 = add i32 %s1, %p2
%a3.s = shl i32 %a, 0
%a3 = ashr i32 %a3.s, 24
%b3.s = shl i32 %b, 0
%b3 = ashr i32 %b3.s, 24
%p3 = mul i32 %a3, %b3
%s3 = add i32 %s2, %p3
ret i32 %s3
}
define internal i32 @recipe.perm(i32 %hi, i32 %lo, i32 %sel) #1 { entry:
%s0.raw = lshr i32 %sel, 0
%s0 = and i32 %s0.raw, 7
%s0.high = icmp uge i32 %s0, 4
%s0.low.shift = mul i32 %s0, 8
%s0.high.index = sub i32 %s0, 4
%s0.high.shift = mul i32 %s0.high.index, 8
%s0.from.lo = lshr i32 %lo, %s0.low.shift
%s0.from.hi = lshr i32 %hi, %s0.high.shift
%s0.picked = select i1 %s0.high, i32 %s0.from.hi, i32 %s0.from.lo
%s0.byte = and i32 %s0.picked, 255
%s0.placed = shl i32 %s0.byte, 0
%o0 = or i32 0, %s0.placed
%s1.raw = lshr i32 %sel, 8
%s1 = and i32 %s1.raw, 7
%s1.high = icmp uge i32 %s1, 4
%s1.low.shift = mul i32 %s1, 8
%s1.high.index = sub i32 %s1, 4
%s1.high.shift = mul i32 %s1.high.index, 8
%s1.from.lo = lshr i32 %lo, %s1.low.shift
%s1.from.hi = lshr i32 %hi, %s1.high.shift
%s1.picked = select i1 %s1.high, i32 %s1.from.hi, i32 %s1.from.lo
%s1.byte = and i32 %s1.picked, 255
%s1.placed = shl i32 %s1.byte, 8
%o1 = or i32 %o0, %s1.placed
%s2.raw = lshr i32 %sel, 16
%s2 = and i32 %s2.raw, 7
%s2.high = icmp uge i32 %s2, 4
%s2.low.shift = mul i32 %s2, 8
%s2.high.index = sub i32 %s2, 4
%s2.high.shift = mul i32 %s2.high.index, 8
%s2.from.lo = lshr i32 %lo, %s2.low.shift
%s2.from.hi = lshr i32 %hi, %s2.high.shift
%s2.picked = select i1 %s2.high, i32 %s2.from.hi, i32 %s2.from.lo
%s2.byte = and i32 %s2.picked, 255
%s2.placed = shl i32 %s2.byte, 16
%o2 = or i32 %o1, %s2.placed
%s3.raw = lshr i32 %sel, 24
%s3 = and i32 %s3.raw, 7
%s3.high = icmp uge i32 %s3, 4
%s3.low.shift = mul i32 %s3, 8
%s3.high.index = sub i32 %s3, 4
%s3.high.shift = mul i32 %s3.high.index, 8
%s3.from.lo = lshr i32 %lo, %s3.low.shift
%s3.from.hi = lshr i32 %hi, %s3.high.shift
%s3.picked = select i1 %s3.high, i32 %s3.from.hi, i32 %s3.from.lo
%s3.byte = and i32 %s3.picked, 255
%s3.placed = shl i32 %s3.byte, 24
%o3 = or i32 %o2, %s3.placed
ret i32 %o3
}
define internal float @recipe.q4k.slice(ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %q8, i32 %slice) #1 { entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.ptr = getelementptr i8, ptr addrspace(1) %block, i64 0
%d.bits = load half, ptr addrspace(1) %d.ptr, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%dmin.ptr = getelementptr i8, ptr addrspace(1) %block, i64 2
%dmin.bits = load half, ptr addrspace(1) %dmin.ptr, align 2
%dmin = call float @recipe.state.from.f16(half %dmin.bits)
%group = udiv i32 %slice, 2
%group.wide = zext i32 %group to i64
%half = and i32 %slice, 1
%group.low = icmp ult i32 %group, 4
%group.plus = add i32 %group, 4
%group.minus = sub i32 %group, 4
%scale.index = select i1 %group.low, i32 %group, i32 %group.plus
%other.index = select i1 %group.low, i32 %group.plus, i32 %group.minus
%scale.index.wide = zext i32 %scale.index to i64
%other.index.wide = zext i32 %other.index to i64
%scale.offset = add i64 4, %scale.index.wide
%other.offset = add i64 4, %other.index.wide
%scale.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %scale.offset
%scale.byte = load i8, ptr addrspace(1) %scale.ptr, align 1
%other.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %other.offset
%other.byte = load i8, ptr addrspace(1) %other.ptr, align 1
%scale.raw = zext i8 %scale.byte to i32
%other.raw = zext i8 %other.byte to i32
%scale.low = and i32 %scale.raw, 63
%minimum.low = and i32 %other.raw, 63
%scale.high.low = and i32 %scale.raw, 15
%minimum.high.low = lshr i32 %scale.raw, 4
%high.bits = and i32 %other.raw, 192
%high = lshr i32 %high.bits, 2
%scale.high = or i32 %scale.high.low, %high
%minimum.high.offset = add i64 4, %group.wide
%minimum.high.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %minimum.high.offset
%minimum.high.byte = load i8, ptr addrspace(1) %minimum.high.ptr, align 1
%minimum.high.raw = zext i8 %minimum.high.byte to i32
%minimum.high.bits = and i32 %minimum.high.raw, 192
%minimum.high.top = lshr i32 %minimum.high.bits, 2
%minimum.high = or i32 %minimum.high.low, %minimum.high.top
%minimum = select i1 %group.low, i32 %minimum.low, i32 %minimum.high
%scale.code = select i1 %group.low, i32 %scale.low, i32 %scale.high
%scale.value = call float @recipe.state.from.u32(i32 %scale.code)
%minimum.value = call float @recipe.state.from.u32(i32 %minimum)
%group.d = call float @recipe.state.mul(float %d, float %scale.value)
%group.dmin = call float @recipe.state.mul(float %dmin, float %minimum.value)
%pair = udiv i32 %group, 2
%pair.wide = zext i32 %pair to i64
%q4.base.part = mul i64 %pair.wide, 32
%q4.base = add i64 %q4.base.part, 16
%group.shift = and i32 %group, 1
%q4.shift = mul i32 %group.shift, 4
%half.wide = zext i32 %half to i64
%q4.half.offset = mul i64 %half.wide, 16
%q4.base.half = add i64 %q4.base, %q4.half.offset
%q8.half.offset = mul i64 %half.wide, 16
%q8.group = zext i32 %group to i64
%q8.offset = mul i64 %q8.group, 36
%q8.block = getelementptr i8, ptr addrspace(3) %q8, i64 %q8.offset
%q8.d.state = load float, ptr addrspace(3) %q8.block, align 4
%dot.zero = add i32 0, 0
%q8.zero = add i32 0, 0
%w0.q4.offset = add i64 %q4.base.half, 0
%w0.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w0.q4.offset
%w0.q4.word = load i32, ptr addrspace(1) %w0.q4.ptr, align 2
%w0.q4.shifted = lshr i32 %w0.q4.word, %q4.shift
%w0.q4.codes = and i32 %w0.q4.shifted, 252645135
%w0.q8.offset = add i64 %q8.half.offset, 4
%w0.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w0.q8.offset
%w0.q8.word = load i32, ptr addrspace(3) %w0.q8.ptr, align 4
%w0.dot = call i32 @recipe.dot4.su(i32 %w0.q4.codes, i32 %w0.q8.word)
%w0.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w0.q8.word)
%w0.dot.sum = add i32 %dot.zero, %w0.dot
%w0.q8.sum = add i32 %q8.zero, %w0.sum
%w1.q4.offset = add i64 %q4.base.half, 4
%w1.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w1.q4.offset
%w1.q4.word = load i32, ptr addrspace(1) %w1.q4.ptr, align 2
%w1.q4.shifted = lshr i32 %w1.q4.word, %q4.shift
%w1.q4.codes = and i32 %w1.q4.shifted, 252645135
%w1.q8.offset = add i64 %q8.half.offset, 8
%w1.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w1.q8.offset
%w1.q8.word = load i32, ptr addrspace(3) %w1.q8.ptr, align 4
%w1.dot = call i32 @recipe.dot4.su(i32 %w1.q4.codes, i32 %w1.q8.word)
%w1.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w1.q8.word)
%w1.dot.sum = add i32 %w0.dot.sum, %w1.dot
%w1.q8.sum = add i32 %w0.q8.sum, %w1.sum
%w2.q4.offset = add i64 %q4.base.half, 8
%w2.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w2.q4.offset
%w2.q4.word = load i32, ptr addrspace(1) %w2.q4.ptr, align 2
%w2.q4.shifted = lshr i32 %w2.q4.word, %q4.shift
%w2.q4.codes = and i32 %w2.q4.shifted, 252645135
%w2.q8.offset = add i64 %q8.half.offset, 12
%w2.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w2.q8.offset
%w2.q8.word = load i32, ptr addrspace(3) %w2.q8.ptr, align 4
%w2.dot = call i32 @recipe.dot4.su(i32 %w2.q4.codes, i32 %w2.q8.word)
%w2.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w2.q8.word)
%w2.dot.sum = add i32 %w1.dot.sum, %w2.dot
%w2.q8.sum = add i32 %w1.q8.sum, %w2.sum
%w3.q4.offset = add i64 %q4.base.half, 12
%w3.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w3.q4.offset
%w3.q4.word = load i32, ptr addrspace(1) %w3.q4.ptr, align 2
%w3.q4.shifted = lshr i32 %w3.q4.word, %q4.shift
%w3.q4.codes = and i32 %w3.q4.shifted, 252645135
%w3.q8.offset = add i64 %q8.half.offset, 16
%w3.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w3.q8.offset
%w3.q8.word = load i32, ptr addrspace(3) %w3.q8.ptr, align 4
%w3.dot = call i32 @recipe.dot4.su(i32 %w3.q4.codes, i32 %w3.q8.word)
%w3.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w3.q8.word)
%w3.dot.sum = add i32 %w2.dot.sum, %w3.dot
%w3.q8.sum = add i32 %w2.q8.sum, %w3.sum
%dot.value = call float @recipe.state.from.s32(i32 %w3.dot.sum)
%q8.value = call float @recipe.state.from.s32(i32 %w3.q8.sum)
%dot.scaled = call float @recipe.state.mul(float %group.d, float %dot.value)
%minimum.scaled = call float @recipe.state.mul(float %group.dmin, float %q8.value)
%value = call float @recipe.state.sub(float %dot.scaled, float %minimum.scaled)
%result = call float @recipe.state.mul(float %q8.d.state, float %value)
ret float %result
}
define internal float @recipe.q6k.slice(ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %q8, i32 %slice) #1 { entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.ptr = getelementptr i8, ptr addrspace(1) %block, i64 208
%d.bits = load half, ptr addrspace(1) %d.ptr, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%chunk = udiv i32 %slice, 8
%group = udiv i32 %slice, 2
%local = urem i32 %group, 4
%half = and i32 %slice, 1
%chunk.wide = zext i32 %chunk to i64
%local.wide = zext i32 %local to i64
%half.wide = zext i32 %half to i64
%low.group = and i32 %local, 1
%low.group.wide = zext i32 %low.group to i64
%ql.extra = mul i64 %low.group.wide, 32
%ql.base.part = mul i64 %chunk.wide, 64
%ql.base = add i64 %ql.base.part, %ql.extra
%qh.base.part = mul i64 %chunk.wide, 32
%qh.base = add i64 %qh.base.part, 128
%ql.shift.group = udiv i32 %local, 2
%ql.shift = mul i32 %ql.shift.group, 4
%qh.shift = mul i32 %local, 2
%q8.group = zext i32 %group to i64
%q8.offset = mul i64 %q8.group, 36
%q8.block = getelementptr i8, ptr addrspace(3) %q8, i64 %q8.offset
%q8.d.state = load float, ptr addrspace(3) %q8.block, align 4
%scale.offset = add i32 %slice, 192
%scale.offset.wide = zext i32 %scale.offset to i64
%scale.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %scale.offset.wide
%scale.byte = load i8, ptr addrspace(1) %scale.ptr, align 1
%scale = sext i8 %scale.byte to i32
%scale.state = call float @recipe.state.from.s32(i32 %scale)
%half.offset = mul i64 %half.wide, 16
%half.ql.base = add i64 %ql.base, %half.offset
%half.qh.base = add i64 %qh.base, %half.offset
%q8.half.offset = mul i64 %half.wide, 16
%dot.zero = add i32 0, 0
%q8.zero = add i32 0, 0
%w0.ql.offset = add i64 %half.ql.base, 0
%w0.ql.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w0.ql.offset
%w0.ql = load i32, ptr addrspace(1) %w0.ql.ptr, align 2
%w0.ql.shifted = lshr i32 %w0.ql, %ql.shift
%w0.ql.codes = and i32 %w0.ql.shifted, 252645135
%w0.qh.offset = add i64 %half.qh.base, 0
%w0.qh.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w0.qh.offset
%w0.qh = load i32, ptr addrspace(1) %w0.qh.ptr, align 2
%w0.qh.shifted = lshr i32 %w0.qh, %qh.shift
%w0.qh.codes = and i32 %w0.qh.shifted, 50529027
%w0.qh.bits = shl i32 %w0.qh.codes, 4
%w0.codes = or i32 %w0.ql.codes, %w0.qh.bits
%w0.q8.offset = add i64 %q8.half.offset, 4
%w0.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w0.q8.offset
%w0.q8 = load i32, ptr addrspace(3) %w0.q8.ptr, align 4
%w0.dot.raw = call i32 @recipe.dot4.su(i32 %w0.codes, i32 %w0.q8)
%w0.sum.raw = call i32 @recipe.dot4.su(i32 16843009, i32 %w0.q8)
%w0.dot.sum = add i32 %dot.zero, %w0.dot.raw
%w0.q8.sum = add i32 %q8.zero, %w0.sum.raw
%w1.ql.offset = add i64 %half.ql.base, 4
%w1.ql.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w1.ql.offset
%w1.ql = load i32, ptr addrspace(1) %w1.ql.ptr, align 2
%w1.ql.shifted = lshr i32 %w1.ql, %ql.shift
%w1.ql.codes = and i32 %w1.ql.shifted, 252645135
%w1.qh.offset = add i64 %half.qh.base, 4
%w1.qh.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w1.qh.offset
%w1.qh = load i32, ptr addrspace(1) %w1.qh.ptr, align 2
%w1.qh.shifted = lshr i32 %w1.qh, %qh.shift
%w1.qh.codes = and i32 %w1.qh.shifted, 50529027
%w1.qh.bits = shl i32 %w1.qh.codes, 4
%w1.codes = or i32 %w1.ql.codes, %w1.qh.bits
%w1.q8.offset = add i64 %q8.half.offset, 8
%w1.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w1.q8.offset
%w1.q8 = load i32, ptr addrspace(3) %w1.q8.ptr, align 4
%w1.dot.raw = call i32 @recipe.dot4.su(i32 %w1.codes, i32 %w1.q8)
%w1.sum.raw = call i32 @recipe.dot4.su(i32 16843009, i32 %w1.q8)
%w1.dot.sum = add i32 %w0.dot.sum, %w1.dot.raw
%w1.q8.sum = add i32 %w0.q8.sum, %w1.sum.raw
%w2.ql.offset = add i64 %half.ql.base, 8
%w2.ql.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w2.ql.offset
%w2.ql = load i32, ptr addrspace(1) %w2.ql.ptr, align 2
%w2.ql.shifted = lshr i32 %w2.ql, %ql.shift
%w2.ql.codes = and i32 %w2.ql.shifted, 252645135
%w2.qh.offset = add i64 %half.qh.base, 8
%w2.qh.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w2.qh.offset
%w2.qh = load i32, ptr addrspace(1) %w2.qh.ptr, align 2
%w2.qh.shifted = lshr i32 %w2.qh, %qh.shift
%w2.qh.codes = and i32 %w2.qh.shifted, 50529027
%w2.qh.bits = shl i32 %w2.qh.codes, 4
%w2.codes = or i32 %w2.ql.codes, %w2.qh.bits
%w2.q8.offset = add i64 %q8.half.offset, 12
%w2.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w2.q8.offset
%w2.q8 = load i32, ptr addrspace(3) %w2.q8.ptr, align 4
%w2.dot.raw = call i32 @recipe.dot4.su(i32 %w2.codes, i32 %w2.q8)
%w2.sum.raw = call i32 @recipe.dot4.su(i32 16843009, i32 %w2.q8)
%w2.dot.sum = add i32 %w1.dot.sum, %w2.dot.raw
%w2.q8.sum = add i32 %w1.q8.sum, %w2.sum.raw
%w3.ql.offset = add i64 %half.ql.base, 12
%w3.ql.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w3.ql.offset
%w3.ql = load i32, ptr addrspace(1) %w3.ql.ptr, align 2
%w3.ql.shifted = lshr i32 %w3.ql, %ql.shift
%w3.ql.codes = and i32 %w3.ql.shifted, 252645135
%w3.qh.offset = add i64 %half.qh.base, 12
%w3.qh.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w3.qh.offset
%w3.qh = load i32, ptr addrspace(1) %w3.qh.ptr, align 2
%w3.qh.shifted = lshr i32 %w3.qh, %qh.shift
%w3.qh.codes = and i32 %w3.qh.shifted, 50529027
%w3.qh.bits = shl i32 %w3.qh.codes, 4
%w3.codes = or i32 %w3.ql.codes, %w3.qh.bits
%w3.q8.offset = add i64 %q8.half.offset, 16
%w3.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w3.q8.offset
%w3.q8 = load i32, ptr addrspace(3) %w3.q8.ptr, align 4
%w3.dot.raw = call i32 @recipe.dot4.su(i32 %w3.codes, i32 %w3.q8)
%w3.sum.raw = call i32 @recipe.dot4.su(i32 16843009, i32 %w3.q8)
%w3.dot.sum = add i32 %w2.dot.sum, %w3.dot.raw
%w3.q8.sum = add i32 %w2.q8.sum, %w3.sum.raw
%dot.offset = mul i32 %w3.q8.sum, 32
%dot.signed = sub i32 %w3.dot.sum, %dot.offset
%dot.value = call float @recipe.state.from.s32(i32 %dot.signed)
%scaled = call float @recipe.state.mul(float %scale.state, float %dot.value)
%value = call float @recipe.state.mul(float %q8.d.state, float %scaled)
%result = call float @recipe.state.mul(float %d, float %value)
ret float %result
}
define internal float @recipe.block32.slice(i32 %kind, ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %q8, i32 %slice) #1 { entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.bits = load half, ptr addrspace(1) %block, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%half.wide = zext i32 %slice to i64
%q8.half.offset = mul i64 %half.wide, 16
%q8.d.state = load float, ptr addrspace(3) %q8, align 4
%nibble.shift = mul i32 %slice, 4
%byte.half = mul i64 %half.wide, 16
%is.iq4 = icmp eq i32 %kind, 1
%is.q41 = icmp eq i32 %kind, 3
%is.q80 = icmp eq i32 %kind, 4
%data.q41 = select i1 %is.q41, i64 4, i64 2
%data.base = select i1 %is.q80, i64 %byte.half, i64 0
%data = add i64 %data.q41, %data.base
%m.ptr = getelementptr i8, ptr addrspace(1) %block, i64 2
%m.bits = load half, ptr addrspace(1) %m.ptr, align 2
%m.loaded = call float @recipe.state.from.f16(half %m.bits)
%zero = call float @recipe.state.from.u1(i1 false)
%m = select i1 %is.q41, float %m.loaded, float %zero
%dot.zero = add i32 0, 0
%q8.zero = add i32 0, 0
%w0.offset = add i64 %data, 0
%w0.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w0.offset
%w0.word = load i32, ptr addrspace(1) %w0.ptr, align 2
%w0.shifted = lshr i32 %w0.word, %nibble.shift
%w0.codes = and i32 %w0.shifted, 252645135
%w0.sel = and i32 %w0.codes, 117901063
%w0.lo = call i32 @recipe.perm(i32 4142587343, i32 3215825025, i32 %w0.sel)
%w0.hi = call i32 @recipe.perm(i32 1901675829, i32 639175937, i32 %w0.sel)
%w0.top = lshr i32 %w0.codes, 3
%w0.top.bits = and i32 %w0.top, 16843009
%w0.mask = mul i32 %w0.top.bits, 255
%w0.mask.not = xor i32 %w0.mask, -1
%w0.hi.masked = and i32 %w0.hi, %w0.mask
%w0.lo.masked = and i32 %w0.lo, %w0.mask.not
%w0.levels = or i32 %w0.hi.masked, %w0.lo.masked
%w0.q8.offset = add i64 %q8.half.offset, 4
%w0.q8.ptr = getelementptr i8, ptr addrspace(3) %q8, i64 %w0.q8.offset
%w0.q8 = load i32, ptr addrspace(3) %w0.q8.ptr, align 4
%w0.dot.iq4 = call i32 @recipe.dot4.ss(i32 %w0.levels, i32 %w0.q8)
%w0.dot.q80 = call i32 @recipe.dot4.ss(i32 %w0.word, i32 %w0.q8)
%w0.dot.q4 = call i32 @recipe.dot4.su(i32 %w0.codes, i32 %w0.q8)
%w0.dot.signed = select i1 %is.q80, i32 %w0.dot.q80, i32 %w0.dot.q4
%w0.dot = select i1 %is.iq4, i32 %w0.dot.iq4, i32 %w0.dot.signed
%w0.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w0.q8)
%w0.dot.acc = add i32 %dot.zero, %w0.dot
%w0.q8.acc = add i32 %q8.zero, %w0.sum
%w1.offset = add i64 %data, 4
%w1.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w1.offset
%w1.word = load i32, ptr addrspace(1) %w1.ptr, align 2
%w1.shifted = lshr i32 %w1.word, %nibble.shift
%w1.codes = and i32 %w1.shifted, 252645135
%w1.sel = and i32 %w1.codes, 117901063
%w1.lo = call i32 @recipe.perm(i32 4142587343, i32 3215825025, i32 %w1.sel)
%w1.hi = call i32 @recipe.perm(i32 1901675829, i32 639175937, i32 %w1.sel)
%w1.top = lshr i32 %w1.codes, 3
%w1.top.bits = and i32 %w1.top, 16843009
%w1.mask = mul i32 %w1.top.bits, 255
%w1.mask.not = xor i32 %w1.mask, -1
%w1.hi.masked = and i32 %w1.hi, %w1.mask
%w1.lo.masked = and i32 %w1.lo, %w1.mask.not
%w1.levels = or i32 %w1.hi.masked, %w1.lo.masked
%w1.q8.offset = add i64 %q8.half.offset, 8
%w1.q8.ptr = getelementptr i8, ptr addrspace(3) %q8, i64 %w1.q8.offset
%w1.q8 = load i32, ptr addrspace(3) %w1.q8.ptr, align 4
%w1.dot.iq4 = call i32 @recipe.dot4.ss(i32 %w1.levels, i32 %w1.q8)
%w1.dot.q80 = call i32 @recipe.dot4.ss(i32 %w1.word, i32 %w1.q8)
%w1.dot.q4 = call i32 @recipe.dot4.su(i32 %w1.codes, i32 %w1.q8)
%w1.dot.signed = select i1 %is.q80, i32 %w1.dot.q80, i32 %w1.dot.q4
%w1.dot = select i1 %is.iq4, i32 %w1.dot.iq4, i32 %w1.dot.signed
%w1.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w1.q8)
%w1.dot.acc = add i32 %w0.dot.acc, %w1.dot
%w1.q8.acc = add i32 %w0.q8.acc, %w1.sum
%w2.offset = add i64 %data, 8
%w2.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w2.offset
%w2.word = load i32, ptr addrspace(1) %w2.ptr, align 2
%w2.shifted = lshr i32 %w2.word, %nibble.shift
%w2.codes = and i32 %w2.shifted, 252645135
%w2.sel = and i32 %w2.codes, 117901063
%w2.lo = call i32 @recipe.perm(i32 4142587343, i32 3215825025, i32 %w2.sel)
%w2.hi = call i32 @recipe.perm(i32 1901675829, i32 639175937, i32 %w2.sel)
%w2.top = lshr i32 %w2.codes, 3
%w2.top.bits = and i32 %w2.top, 16843009
%w2.mask = mul i32 %w2.top.bits, 255
%w2.mask.not = xor i32 %w2.mask, -1
%w2.hi.masked = and i32 %w2.hi, %w2.mask
%w2.lo.masked = and i32 %w2.lo, %w2.mask.not
%w2.levels = or i32 %w2.hi.masked, %w2.lo.masked
%w2.q8.offset = add i64 %q8.half.offset, 12
%w2.q8.ptr = getelementptr i8, ptr addrspace(3) %q8, i64 %w2.q8.offset
%w2.q8 = load i32, ptr addrspace(3) %w2.q8.ptr, align 4
%w2.dot.iq4 = call i32 @recipe.dot4.ss(i32 %w2.levels, i32 %w2.q8)
%w2.dot.q80 = call i32 @recipe.dot4.ss(i32 %w2.word, i32 %w2.q8)
%w2.dot.q4 = call i32 @recipe.dot4.su(i32 %w2.codes, i32 %w2.q8)
%w2.dot.signed = select i1 %is.q80, i32 %w2.dot.q80, i32 %w2.dot.q4
%w2.dot = select i1 %is.iq4, i32 %w2.dot.iq4, i32 %w2.dot.signed
%w2.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w2.q8)
%w2.dot.acc = add i32 %w1.dot.acc, %w2.dot
%w2.q8.acc = add i32 %w1.q8.acc, %w2.sum
%w3.offset = add i64 %data, 12
%w3.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w3.offset
%w3.word = load i32, ptr addrspace(1) %w3.ptr, align 2
%w3.shifted = lshr i32 %w3.word, %nibble.shift
%w3.codes = and i32 %w3.shifted, 252645135
%w3.sel = and i32 %w3.codes, 117901063
%w3.lo = call i32 @recipe.perm(i32 4142587343, i32 3215825025, i32 %w3.sel)
%w3.hi = call i32 @recipe.perm(i32 1901675829, i32 639175937, i32 %w3.sel)
%w3.top = lshr i32 %w3.codes, 3
%w3.top.bits = and i32 %w3.top, 16843009
%w3.mask = mul i32 %w3.top.bits, 255
%w3.mask.not = xor i32 %w3.mask, -1
%w3.hi.masked = and i32 %w3.hi, %w3.mask
%w3.lo.masked = and i32 %w3.lo, %w3.mask.not
%w3.levels = or i32 %w3.hi.masked, %w3.lo.masked
%w3.q8.offset = add i64 %q8.half.offset, 16
%w3.q8.ptr = getelementptr i8, ptr addrspace(3) %q8, i64 %w3.q8.offset
%w3.q8 = load i32, ptr addrspace(3) %w3.q8.ptr, align 4
%w3.dot.iq4 = call i32 @recipe.dot4.ss(i32 %w3.levels, i32 %w3.q8)
%w3.dot.q80 = call i32 @recipe.dot4.ss(i32 %w3.word, i32 %w3.q8)
%w3.dot.q4 = call i32 @recipe.dot4.su(i32 %w3.codes, i32 %w3.q8)
%w3.dot.signed = select i1 %is.q80, i32 %w3.dot.q80, i32 %w3.dot.q4
%w3.dot = select i1 %is.iq4, i32 %w3.dot.iq4, i32 %w3.dot.signed
%w3.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w3.q8)
%w3.dot.acc = add i32 %w2.dot.acc, %w3.dot
%w3.q8.acc = add i32 %w2.q8.acc, %w3.sum
%is.q40 = icmp eq i32 %kind, 2
%q8.eight = mul i32 %w3.q8.acc, 8
%dot.q40 = sub i32 %w3.dot.acc, %q8.eight
%dot.final = select i1 %is.q40, i32 %dot.q40, i32 %w3.dot.acc
%dot.value = call float @recipe.state.from.s32(i32 %dot.final)
%q8.value = call float @recipe.state.from.s32(i32 %w3.q8.acc)
%dot.scaled = call float @recipe.state.mul(float %d, float %dot.value)
%minimum.scaled = call float @recipe.state.mul(float %m, float %q8.value)
%value = call float @recipe.state.add(float %dot.scaled, float %minimum.scaled)
%result = call float @recipe.state.mul(float %q8.d.state, float %value)
ret float %result
}
define internal i64 @recipe.q4k.ints(ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %q8, i32 %slice) #1 { entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.ptr = getelementptr i8, ptr addrspace(1) %block, i64 0
%d.bits = load half, ptr addrspace(1) %d.ptr, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%dmin.ptr = getelementptr i8, ptr addrspace(1) %block, i64 2
%dmin.bits = load half, ptr addrspace(1) %dmin.ptr, align 2
%dmin = call float @recipe.state.from.f16(half %dmin.bits)
%group = udiv i32 %slice, 2
%group.wide = zext i32 %group to i64
%half = and i32 %slice, 1
%group.low = icmp ult i32 %group, 4
%group.plus = add i32 %group, 4
%group.minus = sub i32 %group, 4
%scale.index = select i1 %group.low, i32 %group, i32 %group.plus
%other.index = select i1 %group.low, i32 %group.plus, i32 %group.minus
%scale.index.wide = zext i32 %scale.index to i64
%other.index.wide = zext i32 %other.index to i64
%scale.offset = add i64 4, %scale.index.wide
%other.offset = add i64 4, %other.index.wide
%scale.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %scale.offset
%scale.byte = load i8, ptr addrspace(1) %scale.ptr, align 1
%other.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %other.offset
%other.byte = load i8, ptr addrspace(1) %other.ptr, align 1
%scale.raw = zext i8 %scale.byte to i32
%other.raw = zext i8 %other.byte to i32
%scale.low = and i32 %scale.raw, 63
%minimum.low = and i32 %other.raw, 63
%scale.high.low = and i32 %scale.raw, 15
%minimum.high.low = lshr i32 %scale.raw, 4
%high.bits = and i32 %other.raw, 192
%high = lshr i32 %high.bits, 2
%scale.high = or i32 %scale.high.low, %high
%minimum.high.offset = add i64 4, %group.wide
%minimum.high.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %minimum.high.offset
%minimum.high.byte = load i8, ptr addrspace(1) %minimum.high.ptr, align 1
%minimum.high.raw = zext i8 %minimum.high.byte to i32
%minimum.high.bits = and i32 %minimum.high.raw, 192
%minimum.high.top = lshr i32 %minimum.high.bits, 2
%minimum.high = or i32 %minimum.high.low, %minimum.high.top
%minimum = select i1 %group.low, i32 %minimum.low, i32 %minimum.high
%scale.code = select i1 %group.low, i32 %scale.low, i32 %scale.high
%scale.value = call float @recipe.state.from.u32(i32 %scale.code)
%minimum.value = call float @recipe.state.from.u32(i32 %minimum)
%group.d = call float @recipe.state.mul(float %d, float %scale.value)
%group.dmin = call float @recipe.state.mul(float %dmin, float %minimum.value)
%pair = udiv i32 %group, 2
%pair.wide = zext i32 %pair to i64
%q4.base.part = mul i64 %pair.wide, 32
%q4.base = add i64 %q4.base.part, 16
%group.shift = and i32 %group, 1
%q4.shift = mul i32 %group.shift, 4
%half.wide = zext i32 %half to i64
%q4.half.offset = mul i64 %half.wide, 16
%q4.base.half = add i64 %q4.base, %q4.half.offset
%q8.half.offset = mul i64 %half.wide, 16
%q8.group = zext i32 %group to i64
%q8.offset = mul i64 %q8.group, 36
%q8.block = getelementptr i8, ptr addrspace(3) %q8, i64 %q8.offset
%q8.d.state = load float, ptr addrspace(3) %q8.block, align 4
%dot.zero = add i32 0, 0
%q8.zero = add i32 0, 0
%w0.q4.offset = add i64 %q4.base.half, 0
%w0.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w0.q4.offset
%w0.q4.word = load i32, ptr addrspace(1) %w0.q4.ptr, align 2
%w0.q4.shifted = lshr i32 %w0.q4.word, %q4.shift
%w0.q4.codes = and i32 %w0.q4.shifted, 252645135
%w0.q8.offset = add i64 %q8.half.offset, 4
%w0.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w0.q8.offset
%w0.q8.word = load i32, ptr addrspace(3) %w0.q8.ptr, align 4
%w0.dot = call i32 @recipe.dot4.su(i32 %w0.q4.codes, i32 %w0.q8.word)
%w0.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w0.q8.word)
%w0.dot.sum = add i32 %dot.zero, %w0.dot
%w0.q8.sum = add i32 %q8.zero, %w0.sum
%w1.q4.offset = add i64 %q4.base.half, 4
%w1.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w1.q4.offset
%w1.q4.word = load i32, ptr addrspace(1) %w1.q4.ptr, align 2
%w1.q4.shifted = lshr i32 %w1.q4.word, %q4.shift
%w1.q4.codes = and i32 %w1.q4.shifted, 252645135
%w1.q8.offset = add i64 %q8.half.offset, 8
%w1.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w1.q8.offset
%w1.q8.word = load i32, ptr addrspace(3) %w1.q8.ptr, align 4
%w1.dot = call i32 @recipe.dot4.su(i32 %w1.q4.codes, i32 %w1.q8.word)
%w1.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w1.q8.word)
%w1.dot.sum = add i32 %w0.dot.sum, %w1.dot
%w1.q8.sum = add i32 %w0.q8.sum, %w1.sum
%w2.q4.offset = add i64 %q4.base.half, 8
%w2.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w2.q4.offset
%w2.q4.word = load i32, ptr addrspace(1) %w2.q4.ptr, align 2
%w2.q4.shifted = lshr i32 %w2.q4.word, %q4.shift
%w2.q4.codes = and i32 %w2.q4.shifted, 252645135
%w2.q8.offset = add i64 %q8.half.offset, 12
%w2.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w2.q8.offset
%w2.q8.word = load i32, ptr addrspace(3) %w2.q8.ptr, align 4
%w2.dot = call i32 @recipe.dot4.su(i32 %w2.q4.codes, i32 %w2.q8.word)
%w2.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w2.q8.word)
%w2.dot.sum = add i32 %w1.dot.sum, %w2.dot
%w2.q8.sum = add i32 %w1.q8.sum, %w2.sum
%w3.q4.offset = add i64 %q4.base.half, 12
%w3.q4.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %w3.q4.offset
%w3.q4.word = load i32, ptr addrspace(1) %w3.q4.ptr, align 2
%w3.q4.shifted = lshr i32 %w3.q4.word, %q4.shift
%w3.q4.codes = and i32 %w3.q4.shifted, 252645135
%w3.q8.offset = add i64 %q8.half.offset, 16
%w3.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %w3.q8.offset
%w3.q8.word = load i32, ptr addrspace(3) %w3.q8.ptr, align 4
%w3.dot = call i32 @recipe.dot4.su(i32 %w3.q4.codes, i32 %w3.q8.word)
%w3.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %w3.q8.word)
%w3.dot.sum = add i32 %w2.dot.sum, %w3.dot
%w3.q8.sum = add i32 %w2.q8.sum, %w3.sum
%ints.dot = mul i32 %scale.code, %w3.dot.sum
%ints.min = mul i32 %minimum, %w3.q8.sum
%ints.dot.wide = sext i32 %ints.dot to i64
%ints.dot.high = shl i64 %ints.dot.wide, 32
%ints.min.wide = zext i32 %ints.min to i64
%ints = or i64 %ints.dot.high, %ints.min.wide
ret i64 %ints
}
define internal i64 @recipe.q6k.part(ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %q8, i32 %slice, i32 %word) #1 { entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.ptr = getelementptr i8, ptr addrspace(1) %block, i64 208
%d.bits = load half, ptr addrspace(1) %d.ptr, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%chunk = udiv i32 %slice, 8
%group = udiv i32 %slice, 2
%local = urem i32 %group, 4
%half = and i32 %slice, 1
%chunk.wide = zext i32 %chunk to i64
%local.wide = zext i32 %local to i64
%half.wide = zext i32 %half to i64
%low.group = and i32 %local, 1
%low.group.wide = zext i32 %low.group to i64
%ql.extra = mul i64 %low.group.wide, 32
%ql.base.part = mul i64 %chunk.wide, 64
%ql.base = add i64 %ql.base.part, %ql.extra
%qh.base.part = mul i64 %chunk.wide, 32
%qh.base = add i64 %qh.base.part, 128
%ql.shift.group = udiv i32 %local, 2
%ql.shift = mul i32 %ql.shift.group, 4
%qh.shift = mul i32 %local, 2
%q8.group = zext i32 %group to i64
%q8.offset = mul i64 %q8.group, 36
%q8.block = getelementptr i8, ptr addrspace(3) %q8, i64 %q8.offset
%q8.d.state = load float, ptr addrspace(3) %q8.block, align 4
%scale.offset = add i32 %slice, 192
%scale.offset.wide = zext i32 %scale.offset to i64
%scale.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %scale.offset.wide
%scale.byte = load i8, ptr addrspace(1) %scale.ptr, align 1
%scale = sext i8 %scale.byte to i32
%scale.state = call float @recipe.state.from.s32(i32 %scale)
%half.offset = mul i64 %half.wide, 16
%half.ql.base = add i64 %ql.base, %half.offset
%half.qh.base = add i64 %qh.base, %half.offset
%q8.half.offset = mul i64 %half.wide, 16
%word.wide = zext i32 %word to i64
%word.offset = mul i64 %word.wide, 4
%p.ql.offset = add i64 %half.ql.base, %word.offset
%p.ql.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %p.ql.offset
%p.ql = load i32, ptr addrspace(1) %p.ql.ptr, align 2
%p.ql.shifted = lshr i32 %p.ql, %ql.shift
%p.ql.codes = and i32 %p.ql.shifted, 252645135
%p.qh.offset = add i64 %half.qh.base, %word.offset
%p.qh.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %p.qh.offset
%p.qh = load i32, ptr addrspace(1) %p.qh.ptr, align 2
%p.qh.shifted = lshr i32 %p.qh, %qh.shift
%p.qh.codes = and i32 %p.qh.shifted, 50529027
%p.qh.bits = shl i32 %p.qh.codes, 4
%p.codes = or i32 %p.ql.codes, %p.qh.bits
%p.q8.offset.base = add i64 %q8.half.offset, 4
%p.q8.offset = add i64 %p.q8.offset.base, %word.offset
%p.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.block, i64 %p.q8.offset
%p.q8 = load i32, ptr addrspace(3) %p.q8.ptr, align 4
%p.dot = call i32 @recipe.dot4.su(i32 %p.codes, i32 %p.q8)
%p.sum = call i32 @recipe.dot4.su(i32 16843009, i32 %p.q8)
%p.dot.scaled = mul i32 %scale, %p.dot
%p.sum.scaled = mul i32 %scale, %p.sum
%p.dot.wide = sext i32 %p.dot.scaled to i64
%p.dot.high = shl i64 %p.dot.wide, 32
%p.sum.wide = zext i32 %p.sum.scaled to i64
%p.ints = or i64 %p.dot.high, %p.sum.wide
ret i64 %p.ints
}
; GLIBC BEGIN
@recipe_glibc_exp2f_tab = private unnamed_addr constant [32 x i64] [i64 4607182418800017408, i64 4607140297302181236, i64 4607100335213349135, i64 4607062579818421073, i64 4607027079437701499, i64 4606993883449571754, i64 4606963042313658936, i64 4606934607594512097, i64 4606908631985796885, i64 4606885169335019979, i64 4606864274668794914, i64 4606846004218661165, i64 4606830415447468583, i64 4606817567076339586, i64 4606807519112221737, i64 4606800332876043653, i64 4606796071031487437, i64 4606794797614391156, i64 4606796578062795143, i64 4606801479247646227, i64 4606809569504174299, i64 4606820918663955941, i64 4606835598087680144, i64 4606853680698631517, i64 4606875241016906669, i64 4606900355194379847, i64 4606929101050434204, i64 4606961558108475497, i64 4606997807633245319, i64 4607037932668951391, i64 4607082018078232794, i64 4607130150581978432]
@recipe_glibc_inv_pio4 = private unnamed_addr constant [24 x i32] [i32 162, i32 41721, i32 10680707, i32 -1560706194, i32 -108827058, i32 -2089923004, i32 1850622997, i32 1313084713, i32 1142237692, i32 355073063, i32 704390999, i32 -64530479, i32 660066805, i32 1473377588, i32 -772459299, i32 -181084736, i32 886948059, i32 -574563486, i32 -1059364203, i32 -614296167, i32 1653971260, i32 -1785119677, i32 -1724103792, i32 1011060801]
@recipe_glibc_sign = private unnamed_addr constant [4 x double] [double 1.0, double -1.0, double -1.0, double 1.0]
define internal float @recipe.glibc.expf(float %x) #1 {
entry:
%bits = bitcast float %x to i32
%top = lshr i32 %bits, 20
%abstop = and i32 %top, 2047
%special = icmp uge i32 %abstop, 1067
br i1 %special, label %special.check, label %main
special.check:
%minus.inf = icmp eq i32 %bits, -8388608
br i1 %minus.inf, label %zero, label %special.nan
zero:
ret float 0.0
special.nan:
%nonfinite = icmp uge i32 %abstop, 2040
br i1 %nonfinite, label %nan, label %special.range
nan:
%doubled = fadd float %x, %x
ret float %doubled
special.range:
%over = fcmp ogt float %x, 0x40562E42E0000000
br i1 %over, label %overflow, label %under.check
overflow:
ret float 0x7FF0000000000000
under.check:
%under = fcmp olt float %x, 0xC059FE3680000000
br i1 %under, label %zero, label %main
main:
%xd = fpext float %x to double
%kd.shifted = call double @llvm.fma.f64(double 0x40471547652B82FE, double %xd, double 0x4338000000000000)
%ki = bitcast double %kd.shifted to i64
%kd = fsub double %kd.shifted, 0x4338000000000000
%neg.kd = fneg double %kd
%r = call double @llvm.fma.f64(double 0x40471547652B82FE, double %xd, double %neg.kd)
%index = and i64 %ki, 31
%tab.ptr = getelementptr inbounds [32 x i64], ptr @recipe_glibc_exp2f_tab, i64 0, i64 %index
%tab = load i64, ptr %tab.ptr, align 8
%scaled = shl i64 %ki, 47
%t = add i64 %tab, %scaled
%s = bitcast i64 %t to double
%z = call double @llvm.fma.f64(double 0x3EBC6AF84B912394, double %r, double 0x3F2EBFCE50FAC4F3)
%r2 = fmul double %r, %r
%y0 = call double @llvm.fma.f64(double 0x3F962E42FF0C52D6, double %r, double 1.0)
%y1 = call double @llvm.fma.f64(double %z, double %r2, double %y0)
%y = fmul double %y1, %s
%result = fptrunc double %y to float
ret float %result
}
define internal float @recipe.glibc.sincosf(float %y, i1 %want.cos) #1 {
entry:
%bits = bitcast float %y to i32
%top = lshr i32 %bits, 20
%abstop = and i32 %top, 2047
%x = fpext float %y to double
%small = icmp ult i32 %abstop, 1012
br i1 %small, label %small.check, label %mid.check
small.check:
%tiny = icmp ult i32 %abstop, 920
br i1 %tiny, label %tiny.exit, label %small.poly
tiny.exit:
%tiny.value = select i1 %want.cos, float 1.0, float %y
ret float %tiny.value
small.poly:
%x2.small = fmul double %x, %x
br i1 %want.cos, label %cos.poly, label %sin.poly
mid.check:
%mid = icmp ult i32 %abstop, 1071
br i1 %mid, label %mid.reduce, label %large.check
mid.reduce:
%r.mid = fmul double %x, 0x41645F306DC9C883
%r.int = fptosi double %r.mid to i32
%r.biased = add i32 %r.int, 8388608
%n.mid = ashr i32 %r.biased, 24
%n.mid.d = sitofp i32 %n.mid to double
%n.mid.neg = fneg double %n.mid.d
%xr.mid = call double @llvm.fma.f64(double %n.mid.neg, double 0x3FF921FB54442D18, double %x)
%table.mid = and i32 %n.mid, 2
%parity.mid = and i32 %n.mid, 1
%sign.mid = and i32 %n.mid, 3
br label %reduced
large.check:
%finite = icmp ult i32 %abstop, 2040
br i1 %finite, label %large.reduce, label %invalid
invalid:
%difference = fsub float %y, %y
%nan = fdiv float %difference, %difference
ret float %nan
large.reduce:
%arr.shift = lshr i32 %bits, 26
%arr.index = and i32 %arr.shift, 15
%shift.raw = lshr i32 %bits, 23
%shift = and i32 %shift.raw, 7
%mantissa.raw = and i32 %bits, 16777215
%mantissa = or i32 %mantissa.raw, 8388608
%xi = shl i32 %mantissa, %shift
%arr.index.wide = zext i32 %arr.index to i64
%arr.index.4 = add i64 %arr.index.wide, 4
%arr.index.8 = add i64 %arr.index.wide, 8
%a0.ptr = getelementptr inbounds [24 x i32], ptr @recipe_glibc_inv_pio4, i64 0, i64 %arr.index.wide
%a4.ptr = getelementptr inbounds [24 x i32], ptr @recipe_glibc_inv_pio4, i64 0, i64 %arr.index.4
%a8.ptr = getelementptr inbounds [24 x i32], ptr @recipe_glibc_inv_pio4, i64 0, i64 %arr.index.8
%a0 = load i32, ptr %a0.ptr, align 4
%a4 = load i32, ptr %a4.ptr, align 4
%a8 = load i32, ptr %a8.ptr, align 4
%res0.narrow = mul i32 %xi, %a0
%res0 = zext i32 %res0.narrow to i64
%xi.wide = zext i32 %xi to i64
%a4.wide = zext i32 %a4 to i64
%a8.wide = zext i32 %a8 to i64
%res1 = mul i64 %xi.wide, %a4.wide
%res2 = mul i64 %xi.wide, %a8.wide
%res2.high = lshr i64 %res2, 32
%res0.high = shl i64 %res0, 32
%res0.joined = or i64 %res2.high, %res0.high
%res0.sum = add i64 %res0.joined, %res1
%biased = add i64 %res0.sum, 2305843009213693952
%n.large.wide = lshr i64 %biased, 62
%n.large.shifted = shl i64 %n.large.wide, 62
%remainder = sub i64 %res0.sum, %n.large.shifted
%remainder.d = sitofp i64 %remainder to double
%xr.large = fmul double %remainder.d, 0x3C1921FB54442D18
%n.large = trunc i64 %n.large.wide to i32
%sign.bit = lshr i32 %bits, 31
%quadrant.large = add i32 %n.large, %sign.bit
%table.large = and i32 %quadrant.large, 2
%parity.large = and i32 %n.large, 1
%sign.large = and i32 %quadrant.large, 3
br label %reduced
reduced:
%xr = phi double [ %xr.mid, %mid.reduce ], [ %xr.large, %large.reduce ]
%table = phi i32 [ %table.mid, %mid.reduce ], [ %table.large, %large.reduce ]
%parity = phi i32 [ %parity.mid, %mid.reduce ], [ %parity.large, %large.reduce ]
%sign.index = phi i32 [ %sign.mid, %mid.reduce ], [ %sign.large, %large.reduce ]
%x2.reduced = fmul double %xr, %xr
%odd = icmp ne i32 %parity, 0
%use.cos = xor i1 %odd, %want.cos
%second = icmp ne i32 %table, 0
br i1 %use.cos, label %cos.poly, label %sin.signed
sin.signed:
%sign.index.wide = zext i32 %sign.index to i64
%sign.ptr = getelementptr inbounds [4 x double], ptr @recipe_glibc_sign, i64 0, i64 %sign.index.wide
%sign = load double, ptr %sign.ptr, align 8
%xs.signed = fmul double %xr, %sign
br label %sin.poly
sin.poly:
%xs = phi double [ %x, %small.poly ], [ %xs.signed, %sin.signed ]
%x2.sin = phi double [ %x2.small, %small.poly ], [ %x2.reduced, %sin.signed ]
%s1v = call double @llvm.fma.f64(double %x2.sin, double 0xBF2994EB3774CF24, double 0x3F81107605230BC4)
%x3 = fmul double %x2.sin, %xs
%x5 = fmul double %x2.sin, %x3
%s = call double @llvm.fma.f64(double %x3, double 0xBFC555545995A603, double %xs)
%sin.result = call double @llvm.fma.f64(double %s1v, double %x5, double %s)
%sin.narrow = fptrunc double %sin.result to float
ret float %sin.narrow
cos.poly:
%x2.cos = phi double [ %x2.small, %small.poly ], [ %x2.reduced, %reduced ]
%negated = phi i1 [ false, %small.poly ], [ %second, %reduced ]
%c0 = select i1 %negated, double -1.0, double 1.0
%c1 = select i1 %negated, double 0x3FDFFFFFFD0C621C, double 0xBFDFFFFFFD0C621C
%c2 = select i1 %negated, double 0xBFA55553E1068F19, double 0x3FA55553E1068F19
%c3 = select i1 %negated, double 0x3F56C087E89A359D, double 0xBF56C087E89A359D
%c4 = select i1 %negated, double 0xBEF99343027BF8C3, double 0x3EF99343027BF8C3
%x4 = fmul double %x2.cos, %x2.cos
%c2v = call double @llvm.fma.f64(double %x2.cos, double %c4, double %c3)
%x6 = fmul double %x2.cos, %x4
%c1v = call double @llvm.fma.f64(double %x2.cos, double %c1, double %c0)
%c = call double @llvm.fma.f64(double %x4, double %c2, double %c1v)
%cos.result = call double @llvm.fma.f64(double %c2v, double %x6, double %c)
%cos.narrow = fptrunc double %cos.result to float
ret float %cos.narrow
}
define internal float @recipe.glibc.sinf(float %y) #1 { entry: %result = call float @recipe.glibc.sincosf(float %y, i1 false) ret float %result }
define internal float @recipe.glibc.cosf(float %y) #1 { entry: %result = call float @recipe.glibc.sincosf(float %y, i1 true) ret float %result }
define internal float @recipe.glibc.tanhf(float %x) #1 {
entry:
%bits = bitcast float %x to i32
%exponent.raw = lshr i32 %bits, 23
%magnitude = shl i32 %bits, 1
%exponent = and i32 %exponent.raw, 255
%nonfinite = icmp eq i32 %exponent, 255
br i1 %nonfinite, label %nonfinite.check, label %finite
nonfinite.check:
%payload = shl i32 %bits, 9
%is.nan = icmp ne i32 %payload, 0
br i1 %is.nan, label %nan, label %infinite
nan:
%doubled = fadd float %x, %x
ret float %doubled
infinite:
%negative = icmp slt i32 %bits, 0
%unit = select i1 %negative, float -1.0, float 1.0
ret float %unit
finite:
%small = icmp ult i32 %exponent, 115
br i1 %small, label %small.check, label %range
small.check:
%tiny = icmp ult i32 %exponent, 102
br i1 %tiny, label %tiny.check, label %cubic
tiny.check:
%zero = icmp eq i32 %magnitude, 0
br i1 %zero, label %identity, label %linear
identity:
ret float %x
linear:
%neg.x = fneg float %x
%abs.x = call float @llvm.fabs.f32(float %x)
%linear.value = call float @llvm.fma.f32(float %neg.x, float %abs.x, float %x)
ret float %linear.value
cubic:
%x2 = fmul float %x, %x
%cubic.scale = fmul float 0xBFD5555560000000, %x2
%cubic.value = call float @llvm.fma.f32(float %x, float %cubic.scale, float %x)
ret float %cubic.value
range:
%saturated = icmp ugt i32 %magnitude, 2183158118
br i1 %saturated, label %saturate, label %rational
saturate:
%one = call float @llvm.copysign.f32(float 1.0, float %x)
%step = call float @llvm.copysign.f32(float 0x3E60000000000000, float %x)
%saturated.value = fsub float %one, %step
ret float %saturated.value
rational:
%z = fpext float %x to double
%z2 = fmul double %z, %z
%z4 = fmul double %z2, %z2
%z8 = fmul double %z4, %z4
%n0.product = fmul double %z2, 0x3FC30877B8B72D33
%n0 = fadd double 0x3FF0000000000000, %n0.product
%n2.product = fmul double %z2, 0x3F14101377ABB729
%n2 = fadd double 0x3F7694AA09AE9E5E, %n2.product
%n4.product = fmul double %z2, 0x3E12533756E546F7
%n4 = fadd double 0x3E9E0392B1DB0018, %n4.product
%n6.product = fmul double %z2, 0x3C9B06BE534182DE
%n6 = fadd double 0x3D6D62E5ABE6AE8A, %n6.product
%n0.z4 = fmul double %z4, %n2
%n0.b = fadd double %n0, %n0.z4
%n4.z4 = fmul double %z4, %n6
%n4.b = fadd double %n4, %n4.z4
%n0.z8 = fmul double %z8, %n4.b
%n0.c = fadd double %n0.b, %n0.z8
%d0.product = fmul double %z2, 0x3FDED99131B0EBEA
%d0 = fadd double 0x3FF0000000000000, %d0.product
%d2.product = fmul double %z2, 0x3F47CBDACA0E9FCC
%d2 = fadd double 0x3FA0D27ED6C95A69, %d2.product
%d4.product = fmul double %z2, 0x3E5A6F707C5C71AB
%d4 = fadd double 0x3EDB4E60B892578E, %d4.product
%d6.product = fmul double %z2, 0x3D0CA8230677AA01
%d6 = fadd double 0x3DC35A8B6E2CD94C, %d6.product
%d0.z4 = fmul double %z4, %d2
%d0.b = fadd double %d0, %d0.z4
%d4.z4 = fmul double %z4, %d6
%d4.b = fadd double %d4, %d4.z4
%d0.z8 = fmul double %z8, %d4.b
%d0.c = fadd double %d0.b, %d0.z8
%numerator = fmul double %z, %n0.c
%ratio = fdiv double %numerator, %d0.c
%result = fptrunc double %ratio to float
ret float %result
}
; GLIBC END
define internal float @recipe.libm.exp(float %value) #1 { entry: %result = call float @recipe.glibc.expf(float %value) ret float %result }
define internal float @recipe.libm.cos(float %value) #1 { entry: %result = call float @recipe.glibc.cosf(float %value) ret float %result }
define internal float @recipe.libm.sin(float %value) #1 { entry: %result = call float @recipe.glibc.sinf(float %value) ret float %result }
define internal float @recipe.libm.tanh(float %value) #1 { entry: %result = call float @recipe.glibc.tanhf(float %value) ret float %result }
define internal float @recipe.libm.log(float %value) #1 { entry: %result = call float @recipe.state.log(float %value) ret float %result }
define internal float @recipe.libm.pow(float %base, float %exponent) #1 { entry: %log = call float @recipe.state.log(float %base) %scaled = call float @recipe.state.mul(float %log, float %exponent) %result = call float @recipe.state.exp(float %scaled) ret float %result }

define internal float @recipe.block32.fp32(i32 %kind, ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %tile, i32 %column, i32 %pitch) #1 {
check:
%valid = icmp eq i32 %kind, 4
br i1 %valid, label %entry, label %invalid
invalid:
call void @llvm.trap()
unreachable
entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.bits = load half, ptr addrspace(1) %block, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%weight.codes = getelementptr i8, ptr addrspace(1) %block, i64 2
%base = mul i32 %column, 32
%zero = call float @recipe.state.from.u1(i1 false)
br label %loop
loop:
%i = phi i32 [ 0, %entry ], [ %next, %step ]
%sum = phi float [ %zero, %entry ], [ %added, %step ]
%more = icmp ult i32 %i, 32
br i1 %more, label %step, label %done
step:
%c = add i32 %base, %i
%within = and i32 %c, 15
%quad = lshr i32 %within, 2
%lane = and i32 %c, 3
%group = lshr i32 %c, 4
%row = mul i32 %quad, %pitch
%group.offset = mul i32 %group, 4
%row.group = add i32 %row, %group.offset
%slot = add i32 %row.group, %lane
%ap = getelementptr i8, ptr addrspace(3) %tile, i32 %slot
%a = load i8, ptr addrspace(3) %ap, align 1
%value = call float @recipe.decode(i8 %a)
%wp = getelementptr i8, ptr addrspace(1) %weight.codes, i32 %i
%w = load i8, ptr addrspace(1) %wp, align 1
%wi = sext i8 %w to i32
%weight = call float @recipe.state.from.s32(i32 %wi)
%product = call float @recipe.state.mul(float %weight, float %value)
%added = call float @recipe.state.add(float %sum, float %product)
%next = add i32 %i, 1
br label %loop
done:
%result = call float @recipe.state.mul(float %d, float %sum)
ret float %result
}
define internal float @recipe.block32.i16(i32 %kind, ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %activation, i32 %slice) #1 {
check:
%valid = icmp eq i32 %kind, 4
br i1 %valid, label %entry, label %invalid
invalid:
call void @llvm.trap()
unreachable
entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.bits = load half, ptr addrspace(1) %block, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%s = load float, ptr addrspace(3) %activation, align 4
%codes = getelementptr i8, ptr addrspace(3) %activation, i64 4
%weight.codes = getelementptr i8, ptr addrspace(1) %block, i64 2
br label %loop
loop:
%i = phi i32 [ 0, %entry ], [ %next, %step ]
%sum = phi i32 [ 0, %entry ], [ %added, %step ]
%more = icmp ult i32 %i, 32
br i1 %more, label %step, label %done
step:
%wp = getelementptr i8, ptr addrspace(1) %weight.codes, i32 %i
%w = load i8, ptr addrspace(1) %wp, align 1
%ap = getelementptr i16, ptr addrspace(3) %codes, i32 %i
%a = load i16, ptr addrspace(3) %ap, align 2
%wi = sext i8 %w to i32
%ai = sext i16 %a to i32
%product = mul i32 %wi, %ai
%added = add i32 %sum, %product
%next = add i32 %i, 1
br label %loop
done:
%integer = call float @recipe.state.from.s32(i32 %sum)
%scaled = call float @recipe.state.mul(float %d, float %integer)
%result = call float @recipe.state.mul(float %scaled, float %s)
ret float %result
}
@recipe_block32_iq4_levels = private unnamed_addr constant [16 x i8] [i8 -127, i8 -104, i8 -83, i8 -65, i8 -49, i8 -35, i8 -22, i8 -10, i8 1, i8 13, i8 25, i8 38, i8 53, i8 69, i8 89, i8 113]
declare i32 @llvm.fshr.i32(i32, i32, i32)
define internal float @recipe.q4k.exact(ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %tile, i32 %column, i32 %pitch, i32 %slice) #1 {
entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.bits = load half, ptr addrspace(1) %block, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%dmin.ptr = getelementptr i8, ptr addrspace(1) %block, i64 2
%dmin.bits = load half, ptr addrspace(1) %dmin.ptr, align 2
%dmin = call float @recipe.state.from.f16(half %dmin.bits)
%group = udiv i32 %slice, 2
%group.wide = zext i32 %group to i64
%half = and i32 %slice, 1
%group.low = icmp ult i32 %group, 4
%group.plus = add i32 %group, 4
%group.minus = sub i32 %group, 4
%scale.index = select i1 %group.low, i32 %group, i32 %group.plus
%other.index = select i1 %group.low, i32 %group.plus, i32 %group.minus
%scale.index.wide = zext i32 %scale.index to i64
%other.index.wide = zext i32 %other.index to i64
%scale.offset = add i64 4, %scale.index.wide
%other.offset = add i64 4, %other.index.wide
%scale.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %scale.offset
%scale.byte = load i8, ptr addrspace(1) %scale.ptr, align 1
%other.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %other.offset
%other.byte = load i8, ptr addrspace(1) %other.ptr, align 1
%scale.raw = zext i8 %scale.byte to i32
%other.raw = zext i8 %other.byte to i32
%scale.low = and i32 %scale.raw, 63
%minimum.low = and i32 %other.raw, 63
%scale.high.low = and i32 %scale.raw, 15
%minimum.high.low = lshr i32 %scale.raw, 4
%high.bits = and i32 %other.raw, 192
%high = lshr i32 %high.bits, 2
%scale.high = or i32 %scale.high.low, %high
%minimum.high.offset = add i64 4, %group.wide
%minimum.high.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %minimum.high.offset
%minimum.high.byte = load i8, ptr addrspace(1) %minimum.high.ptr, align 1
%minimum.high.raw = zext i8 %minimum.high.byte to i32
%minimum.high.bits = and i32 %minimum.high.raw, 192
%minimum.high.top = lshr i32 %minimum.high.bits, 2
%minimum.high = or i32 %minimum.high.low, %minimum.high.top
%minimum = select i1 %group.low, i32 %minimum.low, i32 %minimum.high
%scale.code = select i1 %group.low, i32 %scale.low, i32 %scale.high
%scale.value = call float @recipe.state.from.u32(i32 %scale.code)
%minimum.value = call float @recipe.state.from.u32(i32 %minimum)
%group.d = call float @recipe.state.mul(float %d, float %scale.value)
%group.dmin = call float @recipe.state.mul(float %dmin, float %minimum.value)
%pair = udiv i32 %group, 2
%pair.wide = zext i32 %pair to i64
%q4.base.part = mul i64 %pair.wide, 32
%q4.base = add i64 %q4.base.part, 16
%group.shift = and i32 %group, 1
%q4.shift = mul i32 %group.shift, 4
%half.wide = zext i32 %half to i64
%q4.half.offset = mul i64 %half.wide, 16
%q4.base.half = add i64 %q4.base, %q4.half.offset
%column.quad = mul i32 %column, 4
%xq0.row = mul i32 %pitch, 0
%xq0.slot = add i32 %xq0.row, %column.quad
%xq0.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq0.slot
%xq0 = load <4 x i8>, ptr addrspace(3) %xq0.ptr, align 4
%x0.model = extractelement <4 x i8> %xq0, i32 0
%x0 = call float @recipe.decode(i8 %x0.model)
%x1.model = extractelement <4 x i8> %xq0, i32 1
%x1 = call float @recipe.decode(i8 %x1.model)
%x2.model = extractelement <4 x i8> %xq0, i32 2
%x2 = call float @recipe.decode(i8 %x2.model)
%x3.model = extractelement <4 x i8> %xq0, i32 3
%x3 = call float @recipe.decode(i8 %x3.model)
%xq1.row = mul i32 %pitch, 1
%xq1.slot = add i32 %xq1.row, %column.quad
%xq1.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq1.slot
%xq1 = load <4 x i8>, ptr addrspace(3) %xq1.ptr, align 4
%x4.model = extractelement <4 x i8> %xq1, i32 0
%x4 = call float @recipe.decode(i8 %x4.model)
%x5.model = extractelement <4 x i8> %xq1, i32 1
%x5 = call float @recipe.decode(i8 %x5.model)
%x6.model = extractelement <4 x i8> %xq1, i32 2
%x6 = call float @recipe.decode(i8 %x6.model)
%x7.model = extractelement <4 x i8> %xq1, i32 3
%x7 = call float @recipe.decode(i8 %x7.model)
%xq2.row = mul i32 %pitch, 2
%xq2.slot = add i32 %xq2.row, %column.quad
%xq2.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq2.slot
%xq2 = load <4 x i8>, ptr addrspace(3) %xq2.ptr, align 4
%x8.model = extractelement <4 x i8> %xq2, i32 0
%x8 = call float @recipe.decode(i8 %x8.model)
%x9.model = extractelement <4 x i8> %xq2, i32 1
%x9 = call float @recipe.decode(i8 %x9.model)
%x10.model = extractelement <4 x i8> %xq2, i32 2
%x10 = call float @recipe.decode(i8 %x10.model)
%x11.model = extractelement <4 x i8> %xq2, i32 3
%x11 = call float @recipe.decode(i8 %x11.model)
%xq3.row = mul i32 %pitch, 3
%xq3.slot = add i32 %xq3.row, %column.quad
%xq3.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq3.slot
%xq3 = load <4 x i8>, ptr addrspace(3) %xq3.ptr, align 4
%x12.model = extractelement <4 x i8> %xq3, i32 0
%x12 = call float @recipe.decode(i8 %x12.model)
%x13.model = extractelement <4 x i8> %xq3, i32 1
%x13 = call float @recipe.decode(i8 %x13.model)
%x14.model = extractelement <4 x i8> %xq3, i32 2
%x14 = call float @recipe.decode(i8 %x14.model)
%x15.model = extractelement <4 x i8> %xq3, i32 3
%x15 = call float @recipe.decode(i8 %x15.model)
%qw.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %q4.base.half
%qw = load <4 x i32>, ptr addrspace(1) %qw.ptr, align 16
%qw0 = extractelement <4 x i32> %qw, i32 0
%q0.byte.shift = lshr i32 %qw0, 0
%q0.raw = and i32 %q0.byte.shift, 255
%q1.byte.shift = lshr i32 %qw0, 8
%q1.raw = and i32 %q1.byte.shift, 255
%q2.byte.shift = lshr i32 %qw0, 16
%q2.raw = and i32 %q2.byte.shift, 255
%q3.byte.shift = lshr i32 %qw0, 24
%q3.raw = and i32 %q3.byte.shift, 255
%qw1 = extractelement <4 x i32> %qw, i32 1
%q4.byte.shift = lshr i32 %qw1, 0
%q4.raw = and i32 %q4.byte.shift, 255
%q5.byte.shift = lshr i32 %qw1, 8
%q5.raw = and i32 %q5.byte.shift, 255
%q6.byte.shift = lshr i32 %qw1, 16
%q6.raw = and i32 %q6.byte.shift, 255
%q7.byte.shift = lshr i32 %qw1, 24
%q7.raw = and i32 %q7.byte.shift, 255
%qw2 = extractelement <4 x i32> %qw, i32 2
%q8.byte.shift = lshr i32 %qw2, 0
%q8.raw = and i32 %q8.byte.shift, 255
%q9.byte.shift = lshr i32 %qw2, 8
%q9.raw = and i32 %q9.byte.shift, 255
%q10.byte.shift = lshr i32 %qw2, 16
%q10.raw = and i32 %q10.byte.shift, 255
%q11.byte.shift = lshr i32 %qw2, 24
%q11.raw = and i32 %q11.byte.shift, 255
%qw3 = extractelement <4 x i32> %qw, i32 3
%q12.byte.shift = lshr i32 %qw3, 0
%q12.raw = and i32 %q12.byte.shift, 255
%q13.byte.shift = lshr i32 %qw3, 8
%q13.raw = and i32 %q13.byte.shift, 255
%q14.byte.shift = lshr i32 %qw3, 16
%q14.raw = and i32 %q14.byte.shift, 255
%q15.byte.shift = lshr i32 %qw3, 24
%q15.raw = and i32 %q15.byte.shift, 255
%acc0.0 = call float @recipe.state.from.u1(i1 false)
%xsum0.0 = call float @recipe.state.from.u1(i1 false)
%acc1.0 = call float @recipe.state.from.u1(i1 false)
%xsum1.0 = call float @recipe.state.from.u1(i1 false)
%acc2.0 = call float @recipe.state.from.u1(i1 false)
%xsum2.0 = call float @recipe.state.from.u1(i1 false)
%acc3.0 = call float @recipe.state.from.u1(i1 false)
%xsum3.0 = call float @recipe.state.from.u1(i1 false)
%q0.shifted = lshr i32 %q0.raw, %q4.shift
%code0 = and i32 %q0.shifted, 15
%code0.state = call float @recipe.state.from.s32(i32 %code0)
%acc0.1 = call float @recipe.state.madd(float %acc0.0, float %code0.state, float %x0)
%xsum0.1 = call float @recipe.state.add(float %xsum0.0, float %x0)
%q1.shifted = lshr i32 %q1.raw, %q4.shift
%code1 = and i32 %q1.shifted, 15
%code1.state = call float @recipe.state.from.s32(i32 %code1)
%acc1.1 = call float @recipe.state.madd(float %acc1.0, float %code1.state, float %x1)
%xsum1.1 = call float @recipe.state.add(float %xsum1.0, float %x1)
%q2.shifted = lshr i32 %q2.raw, %q4.shift
%code2 = and i32 %q2.shifted, 15
%code2.state = call float @recipe.state.from.s32(i32 %code2)
%acc2.1 = call float @recipe.state.madd(float %acc2.0, float %code2.state, float %x2)
%xsum2.1 = call float @recipe.state.add(float %xsum2.0, float %x2)
%q3.shifted = lshr i32 %q3.raw, %q4.shift
%code3 = and i32 %q3.shifted, 15
%code3.state = call float @recipe.state.from.s32(i32 %code3)
%acc3.1 = call float @recipe.state.madd(float %acc3.0, float %code3.state, float %x3)
%xsum3.1 = call float @recipe.state.add(float %xsum3.0, float %x3)
%q4.shifted = lshr i32 %q4.raw, %q4.shift
%code4 = and i32 %q4.shifted, 15
%code4.state = call float @recipe.state.from.s32(i32 %code4)
%acc0.2 = call float @recipe.state.madd(float %acc0.1, float %code4.state, float %x4)
%xsum0.2 = call float @recipe.state.add(float %xsum0.1, float %x4)
%q5.shifted = lshr i32 %q5.raw, %q4.shift
%code5 = and i32 %q5.shifted, 15
%code5.state = call float @recipe.state.from.s32(i32 %code5)
%acc1.2 = call float @recipe.state.madd(float %acc1.1, float %code5.state, float %x5)
%xsum1.2 = call float @recipe.state.add(float %xsum1.1, float %x5)
%q6.shifted = lshr i32 %q6.raw, %q4.shift
%code6 = and i32 %q6.shifted, 15
%code6.state = call float @recipe.state.from.s32(i32 %code6)
%acc2.2 = call float @recipe.state.madd(float %acc2.1, float %code6.state, float %x6)
%xsum2.2 = call float @recipe.state.add(float %xsum2.1, float %x6)
%q7.shifted = lshr i32 %q7.raw, %q4.shift
%code7 = and i32 %q7.shifted, 15
%code7.state = call float @recipe.state.from.s32(i32 %code7)
%acc3.2 = call float @recipe.state.madd(float %acc3.1, float %code7.state, float %x7)
%xsum3.2 = call float @recipe.state.add(float %xsum3.1, float %x7)
%q8.shifted = lshr i32 %q8.raw, %q4.shift
%code8 = and i32 %q8.shifted, 15
%code8.state = call float @recipe.state.from.s32(i32 %code8)
%acc0.3 = call float @recipe.state.madd(float %acc0.2, float %code8.state, float %x8)
%xsum0.3 = call float @recipe.state.add(float %xsum0.2, float %x8)
%q9.shifted = lshr i32 %q9.raw, %q4.shift
%code9 = and i32 %q9.shifted, 15
%code9.state = call float @recipe.state.from.s32(i32 %code9)
%acc1.3 = call float @recipe.state.madd(float %acc1.2, float %code9.state, float %x9)
%xsum1.3 = call float @recipe.state.add(float %xsum1.2, float %x9)
%q10.shifted = lshr i32 %q10.raw, %q4.shift
%code10 = and i32 %q10.shifted, 15
%code10.state = call float @recipe.state.from.s32(i32 %code10)
%acc2.3 = call float @recipe.state.madd(float %acc2.2, float %code10.state, float %x10)
%xsum2.3 = call float @recipe.state.add(float %xsum2.2, float %x10)
%q11.shifted = lshr i32 %q11.raw, %q4.shift
%code11 = and i32 %q11.shifted, 15
%code11.state = call float @recipe.state.from.s32(i32 %code11)
%acc3.3 = call float @recipe.state.madd(float %acc3.2, float %code11.state, float %x11)
%xsum3.3 = call float @recipe.state.add(float %xsum3.2, float %x11)
%q12.shifted = lshr i32 %q12.raw, %q4.shift
%code12 = and i32 %q12.shifted, 15
%code12.state = call float @recipe.state.from.s32(i32 %code12)
%acc0.4 = call float @recipe.state.madd(float %acc0.3, float %code12.state, float %x12)
%xsum0.4 = call float @recipe.state.add(float %xsum0.3, float %x12)
%q13.shifted = lshr i32 %q13.raw, %q4.shift
%code13 = and i32 %q13.shifted, 15
%code13.state = call float @recipe.state.from.s32(i32 %code13)
%acc1.4 = call float @recipe.state.madd(float %acc1.3, float %code13.state, float %x13)
%xsum1.4 = call float @recipe.state.add(float %xsum1.3, float %x13)
%q14.shifted = lshr i32 %q14.raw, %q4.shift
%code14 = and i32 %q14.shifted, 15
%code14.state = call float @recipe.state.from.s32(i32 %code14)
%acc2.4 = call float @recipe.state.madd(float %acc2.3, float %code14.state, float %x14)
%xsum2.4 = call float @recipe.state.add(float %xsum2.3, float %x14)
%q15.shifted = lshr i32 %q15.raw, %q4.shift
%code15 = and i32 %q15.shifted, 15
%code15.state = call float @recipe.state.from.s32(i32 %code15)
%acc3.4 = call float @recipe.state.madd(float %acc3.3, float %code15.state, float %x15)
%xsum3.4 = call float @recipe.state.add(float %xsum3.3, float %x15)
%acc01 = call float @recipe.state.add(float %acc0.4, float %acc1.4)
%acc23 = call float @recipe.state.add(float %acc2.4, float %acc3.4)
%acc16 = call float @recipe.state.add(float %acc01, float %acc23)
%xsum01 = call float @recipe.state.add(float %xsum0.4, float %xsum1.4)
%xsum23 = call float @recipe.state.add(float %xsum2.4, float %xsum3.4)
%xsum16 = call float @recipe.state.add(float %xsum01, float %xsum23)
%dot.scaled = call float @recipe.state.mul(float %group.d, float %acc16)
%minimum.scaled = call float @recipe.state.mul(float %group.dmin, float %xsum16)
%result = call float @recipe.state.sub(float %dot.scaled, float %minimum.scaled)
ret float %result
}
define internal float @recipe.q6k.exact(ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %tile, i32 %column, i32 %pitch, i32 %slice) #1 {
entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.ptr = getelementptr i8, ptr addrspace(1) %block, i64 208
%d.bits = load half, ptr addrspace(1) %d.ptr, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%chunk = udiv i32 %slice, 8
%group = udiv i32 %slice, 2
%local = urem i32 %group, 4
%half = and i32 %slice, 1
%chunk.wide = zext i32 %chunk to i64
%half.wide = zext i32 %half to i64
%low.group = and i32 %local, 1
%low.group.wide = zext i32 %low.group to i64
%ql.extra = mul i64 %low.group.wide, 32
%ql.base.part = mul i64 %chunk.wide, 64
%ql.base = add i64 %ql.base.part, %ql.extra
%qh.base.part = mul i64 %chunk.wide, 32
%qh.base = add i64 %qh.base.part, 128
%ql.shift.group = udiv i32 %local, 2
%ql.shift = mul i32 %ql.shift.group, 4
%qh.shift = mul i32 %local, 2
%scale.offset = add i32 %slice, 192
%scale.offset.wide = zext i32 %scale.offset to i64
%scale.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %scale.offset.wide
%scale.byte = load i8, ptr addrspace(1) %scale.ptr, align 1
%scale = sext i8 %scale.byte to i32
%scale.state = call float @recipe.state.from.s32(i32 %scale)
%half.offset = mul i64 %half.wide, 16
%half.ql.base = add i64 %ql.base, %half.offset
%half.qh.base = add i64 %qh.base, %half.offset
%column.quad = mul i32 %column, 4
%xq0.row = mul i32 %pitch, 0
%xq0.slot = add i32 %xq0.row, %column.quad
%xq0.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq0.slot
%xq0 = load <4 x i8>, ptr addrspace(3) %xq0.ptr, align 4
%x0.model = extractelement <4 x i8> %xq0, i32 0
%x0 = call float @recipe.decode(i8 %x0.model)
%x1.model = extractelement <4 x i8> %xq0, i32 1
%x1 = call float @recipe.decode(i8 %x1.model)
%x2.model = extractelement <4 x i8> %xq0, i32 2
%x2 = call float @recipe.decode(i8 %x2.model)
%x3.model = extractelement <4 x i8> %xq0, i32 3
%x3 = call float @recipe.decode(i8 %x3.model)
%xq1.row = mul i32 %pitch, 1
%xq1.slot = add i32 %xq1.row, %column.quad
%xq1.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq1.slot
%xq1 = load <4 x i8>, ptr addrspace(3) %xq1.ptr, align 4
%x4.model = extractelement <4 x i8> %xq1, i32 0
%x4 = call float @recipe.decode(i8 %x4.model)
%x5.model = extractelement <4 x i8> %xq1, i32 1
%x5 = call float @recipe.decode(i8 %x5.model)
%x6.model = extractelement <4 x i8> %xq1, i32 2
%x6 = call float @recipe.decode(i8 %x6.model)
%x7.model = extractelement <4 x i8> %xq1, i32 3
%x7 = call float @recipe.decode(i8 %x7.model)
%xq2.row = mul i32 %pitch, 2
%xq2.slot = add i32 %xq2.row, %column.quad
%xq2.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq2.slot
%xq2 = load <4 x i8>, ptr addrspace(3) %xq2.ptr, align 4
%x8.model = extractelement <4 x i8> %xq2, i32 0
%x8 = call float @recipe.decode(i8 %x8.model)
%x9.model = extractelement <4 x i8> %xq2, i32 1
%x9 = call float @recipe.decode(i8 %x9.model)
%x10.model = extractelement <4 x i8> %xq2, i32 2
%x10 = call float @recipe.decode(i8 %x10.model)
%x11.model = extractelement <4 x i8> %xq2, i32 3
%x11 = call float @recipe.decode(i8 %x11.model)
%xq3.row = mul i32 %pitch, 3
%xq3.slot = add i32 %xq3.row, %column.quad
%xq3.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq3.slot
%xq3 = load <4 x i8>, ptr addrspace(3) %xq3.ptr, align 4
%x12.model = extractelement <4 x i8> %xq3, i32 0
%x12 = call float @recipe.decode(i8 %x12.model)
%x13.model = extractelement <4 x i8> %xq3, i32 1
%x13 = call float @recipe.decode(i8 %x13.model)
%x14.model = extractelement <4 x i8> %xq3, i32 2
%x14 = call float @recipe.decode(i8 %x14.model)
%x15.model = extractelement <4 x i8> %xq3, i32 3
%x15 = call float @recipe.decode(i8 %x15.model)
%qlw.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %half.ql.base
%qlw.addr = ptrtoint ptr addrspace(1) %qlw.ptr to i64
%qlw.low = and i64 %qlw.addr, 15
%qlw.line.addr = sub i64 %qlw.addr, %qlw.low
%qlw.line = inttoptr i64 %qlw.line.addr to ptr addrspace(1)
%qlw.lo = load <4 x i32>, ptr addrspace(1) %qlw.line, align 16
%qlw.next = getelementptr i8, ptr addrspace(1) %qlw.line, i64 16
%qlw.hi = load <4 x i32>, ptr addrspace(1) %qlw.next, align 16
%qlw.q = lshr i64 %qlw.low, 2
%qlw.q32 = trunc i64 %qlw.q to i32
%qlw.r64 = and i64 %qlw.low, 3
%qlw.r32 = trunc i64 %qlw.r64 to i32
%qlw.r = shl i32 %qlw.r32, 3
%qlin0 = extractelement <4 x i32> %qlw.lo, i32 0
%qlin1 = extractelement <4 x i32> %qlw.lo, i32 1
%qlin2 = extractelement <4 x i32> %qlw.lo, i32 2
%qlin3 = extractelement <4 x i32> %qlw.lo, i32 3
%qlin4 = extractelement <4 x i32> %qlw.hi, i32 0
%qlin5 = extractelement <4 x i32> %qlw.hi, i32 1
%qlin6 = extractelement <4 x i32> %qlw.hi, i32 2
%qlin7 = extractelement <4 x i32> %qlw.hi, i32 3
%qlw.is0 = icmp eq i32 %qlw.q32, 0
%qlw.is1 = icmp eq i32 %qlw.q32, 1
%qlw.is2 = icmp eq i32 %qlw.q32, 2
%qlw.is3 = icmp eq i32 %qlw.q32, 3
%qlw0.a.23 = select i1 %qlw.is2, i32 %qlin2, i32 %qlin3
%qlw0.a.123 = select i1 %qlw.is1, i32 %qlin1, i32 %qlw0.a.23
%qlw0.a = select i1 %qlw.is0, i32 %qlin0, i32 %qlw0.a.123
%qlw0.b.23 = select i1 %qlw.is2, i32 %qlin3, i32 %qlin4
%qlw0.b.123 = select i1 %qlw.is1, i32 %qlin2, i32 %qlw0.b.23
%qlw0.b = select i1 %qlw.is0, i32 %qlin1, i32 %qlw0.b.123
%qlw0 = call i32 @llvm.fshr.i32(i32 %qlw0.b, i32 %qlw0.a, i32 %qlw.r)
%qlw1.a.23 = select i1 %qlw.is2, i32 %qlin3, i32 %qlin4
%qlw1.a.123 = select i1 %qlw.is1, i32 %qlin2, i32 %qlw1.a.23
%qlw1.a = select i1 %qlw.is0, i32 %qlin1, i32 %qlw1.a.123
%qlw1.b.23 = select i1 %qlw.is2, i32 %qlin4, i32 %qlin5
%qlw1.b.123 = select i1 %qlw.is1, i32 %qlin3, i32 %qlw1.b.23
%qlw1.b = select i1 %qlw.is0, i32 %qlin2, i32 %qlw1.b.123
%qlw1 = call i32 @llvm.fshr.i32(i32 %qlw1.b, i32 %qlw1.a, i32 %qlw.r)
%qlw2.a.23 = select i1 %qlw.is2, i32 %qlin4, i32 %qlin5
%qlw2.a.123 = select i1 %qlw.is1, i32 %qlin3, i32 %qlw2.a.23
%qlw2.a = select i1 %qlw.is0, i32 %qlin2, i32 %qlw2.a.123
%qlw2.b.23 = select i1 %qlw.is2, i32 %qlin5, i32 %qlin6
%qlw2.b.123 = select i1 %qlw.is1, i32 %qlin4, i32 %qlw2.b.23
%qlw2.b = select i1 %qlw.is0, i32 %qlin3, i32 %qlw2.b.123
%qlw2 = call i32 @llvm.fshr.i32(i32 %qlw2.b, i32 %qlw2.a, i32 %qlw.r)
%qlw3.a.23 = select i1 %qlw.is2, i32 %qlin5, i32 %qlin6
%qlw3.a.123 = select i1 %qlw.is1, i32 %qlin4, i32 %qlw3.a.23
%qlw3.a = select i1 %qlw.is0, i32 %qlin3, i32 %qlw3.a.123
%qlw3.b.23 = select i1 %qlw.is2, i32 %qlin6, i32 %qlin7
%qlw3.b.123 = select i1 %qlw.is1, i32 %qlin5, i32 %qlw3.b.23
%qlw3.b = select i1 %qlw.is0, i32 %qlin4, i32 %qlw3.b.123
%qlw3 = call i32 @llvm.fshr.i32(i32 %qlw3.b, i32 %qlw3.a, i32 %qlw.r)
%ql0.byte.shift = lshr i32 %qlw0, 0
%ql0.raw = and i32 %ql0.byte.shift, 255
%ql1.byte.shift = lshr i32 %qlw0, 8
%ql1.raw = and i32 %ql1.byte.shift, 255
%ql2.byte.shift = lshr i32 %qlw0, 16
%ql2.raw = and i32 %ql2.byte.shift, 255
%ql3.byte.shift = lshr i32 %qlw0, 24
%ql3.raw = and i32 %ql3.byte.shift, 255
%ql4.byte.shift = lshr i32 %qlw1, 0
%ql4.raw = and i32 %ql4.byte.shift, 255
%ql5.byte.shift = lshr i32 %qlw1, 8
%ql5.raw = and i32 %ql5.byte.shift, 255
%ql6.byte.shift = lshr i32 %qlw1, 16
%ql6.raw = and i32 %ql6.byte.shift, 255
%ql7.byte.shift = lshr i32 %qlw1, 24
%ql7.raw = and i32 %ql7.byte.shift, 255
%ql8.byte.shift = lshr i32 %qlw2, 0
%ql8.raw = and i32 %ql8.byte.shift, 255
%ql9.byte.shift = lshr i32 %qlw2, 8
%ql9.raw = and i32 %ql9.byte.shift, 255
%ql10.byte.shift = lshr i32 %qlw2, 16
%ql10.raw = and i32 %ql10.byte.shift, 255
%ql11.byte.shift = lshr i32 %qlw2, 24
%ql11.raw = and i32 %ql11.byte.shift, 255
%ql12.byte.shift = lshr i32 %qlw3, 0
%ql12.raw = and i32 %ql12.byte.shift, 255
%ql13.byte.shift = lshr i32 %qlw3, 8
%ql13.raw = and i32 %ql13.byte.shift, 255
%ql14.byte.shift = lshr i32 %qlw3, 16
%ql14.raw = and i32 %ql14.byte.shift, 255
%ql15.byte.shift = lshr i32 %qlw3, 24
%ql15.raw = and i32 %ql15.byte.shift, 255
%qhw.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %half.qh.base
%qhw.addr = ptrtoint ptr addrspace(1) %qhw.ptr to i64
%qhw.low = and i64 %qhw.addr, 15
%qhw.line.addr = sub i64 %qhw.addr, %qhw.low
%qhw.line = inttoptr i64 %qhw.line.addr to ptr addrspace(1)
%qhw.lo = load <4 x i32>, ptr addrspace(1) %qhw.line, align 16
%qhw.next = getelementptr i8, ptr addrspace(1) %qhw.line, i64 16
%qhw.hi = load <4 x i32>, ptr addrspace(1) %qhw.next, align 16
%qhw.q = lshr i64 %qhw.low, 2
%qhw.q32 = trunc i64 %qhw.q to i32
%qhw.r64 = and i64 %qhw.low, 3
%qhw.r32 = trunc i64 %qhw.r64 to i32
%qhw.r = shl i32 %qhw.r32, 3
%qhin0 = extractelement <4 x i32> %qhw.lo, i32 0
%qhin1 = extractelement <4 x i32> %qhw.lo, i32 1
%qhin2 = extractelement <4 x i32> %qhw.lo, i32 2
%qhin3 = extractelement <4 x i32> %qhw.lo, i32 3
%qhin4 = extractelement <4 x i32> %qhw.hi, i32 0
%qhin5 = extractelement <4 x i32> %qhw.hi, i32 1
%qhin6 = extractelement <4 x i32> %qhw.hi, i32 2
%qhin7 = extractelement <4 x i32> %qhw.hi, i32 3
%qhw.is0 = icmp eq i32 %qhw.q32, 0
%qhw.is1 = icmp eq i32 %qhw.q32, 1
%qhw.is2 = icmp eq i32 %qhw.q32, 2
%qhw.is3 = icmp eq i32 %qhw.q32, 3
%qhw0.a.23 = select i1 %qhw.is2, i32 %qhin2, i32 %qhin3
%qhw0.a.123 = select i1 %qhw.is1, i32 %qhin1, i32 %qhw0.a.23
%qhw0.a = select i1 %qhw.is0, i32 %qhin0, i32 %qhw0.a.123
%qhw0.b.23 = select i1 %qhw.is2, i32 %qhin3, i32 %qhin4
%qhw0.b.123 = select i1 %qhw.is1, i32 %qhin2, i32 %qhw0.b.23
%qhw0.b = select i1 %qhw.is0, i32 %qhin1, i32 %qhw0.b.123
%qhw0 = call i32 @llvm.fshr.i32(i32 %qhw0.b, i32 %qhw0.a, i32 %qhw.r)
%qhw1.a.23 = select i1 %qhw.is2, i32 %qhin3, i32 %qhin4
%qhw1.a.123 = select i1 %qhw.is1, i32 %qhin2, i32 %qhw1.a.23
%qhw1.a = select i1 %qhw.is0, i32 %qhin1, i32 %qhw1.a.123
%qhw1.b.23 = select i1 %qhw.is2, i32 %qhin4, i32 %qhin5
%qhw1.b.123 = select i1 %qhw.is1, i32 %qhin3, i32 %qhw1.b.23
%qhw1.b = select i1 %qhw.is0, i32 %qhin2, i32 %qhw1.b.123
%qhw1 = call i32 @llvm.fshr.i32(i32 %qhw1.b, i32 %qhw1.a, i32 %qhw.r)
%qhw2.a.23 = select i1 %qhw.is2, i32 %qhin4, i32 %qhin5
%qhw2.a.123 = select i1 %qhw.is1, i32 %qhin3, i32 %qhw2.a.23
%qhw2.a = select i1 %qhw.is0, i32 %qhin2, i32 %qhw2.a.123
%qhw2.b.23 = select i1 %qhw.is2, i32 %qhin5, i32 %qhin6
%qhw2.b.123 = select i1 %qhw.is1, i32 %qhin4, i32 %qhw2.b.23
%qhw2.b = select i1 %qhw.is0, i32 %qhin3, i32 %qhw2.b.123
%qhw2 = call i32 @llvm.fshr.i32(i32 %qhw2.b, i32 %qhw2.a, i32 %qhw.r)
%qhw3.a.23 = select i1 %qhw.is2, i32 %qhin5, i32 %qhin6
%qhw3.a.123 = select i1 %qhw.is1, i32 %qhin4, i32 %qhw3.a.23
%qhw3.a = select i1 %qhw.is0, i32 %qhin3, i32 %qhw3.a.123
%qhw3.b.23 = select i1 %qhw.is2, i32 %qhin6, i32 %qhin7
%qhw3.b.123 = select i1 %qhw.is1, i32 %qhin5, i32 %qhw3.b.23
%qhw3.b = select i1 %qhw.is0, i32 %qhin4, i32 %qhw3.b.123
%qhw3 = call i32 @llvm.fshr.i32(i32 %qhw3.b, i32 %qhw3.a, i32 %qhw.r)
%qh0.byte.shift = lshr i32 %qhw0, 0
%qh0.raw = and i32 %qh0.byte.shift, 255
%qh1.byte.shift = lshr i32 %qhw0, 8
%qh1.raw = and i32 %qh1.byte.shift, 255
%qh2.byte.shift = lshr i32 %qhw0, 16
%qh2.raw = and i32 %qh2.byte.shift, 255
%qh3.byte.shift = lshr i32 %qhw0, 24
%qh3.raw = and i32 %qh3.byte.shift, 255
%qh4.byte.shift = lshr i32 %qhw1, 0
%qh4.raw = and i32 %qh4.byte.shift, 255
%qh5.byte.shift = lshr i32 %qhw1, 8
%qh5.raw = and i32 %qh5.byte.shift, 255
%qh6.byte.shift = lshr i32 %qhw1, 16
%qh6.raw = and i32 %qh6.byte.shift, 255
%qh7.byte.shift = lshr i32 %qhw1, 24
%qh7.raw = and i32 %qh7.byte.shift, 255
%qh8.byte.shift = lshr i32 %qhw2, 0
%qh8.raw = and i32 %qh8.byte.shift, 255
%qh9.byte.shift = lshr i32 %qhw2, 8
%qh9.raw = and i32 %qh9.byte.shift, 255
%qh10.byte.shift = lshr i32 %qhw2, 16
%qh10.raw = and i32 %qh10.byte.shift, 255
%qh11.byte.shift = lshr i32 %qhw2, 24
%qh11.raw = and i32 %qh11.byte.shift, 255
%qh12.byte.shift = lshr i32 %qhw3, 0
%qh12.raw = and i32 %qh12.byte.shift, 255
%qh13.byte.shift = lshr i32 %qhw3, 8
%qh13.raw = and i32 %qh13.byte.shift, 255
%qh14.byte.shift = lshr i32 %qhw3, 16
%qh14.raw = and i32 %qh14.byte.shift, 255
%qh15.byte.shift = lshr i32 %qhw3, 24
%qh15.raw = and i32 %qh15.byte.shift, 255
%acc0.0 = call float @recipe.state.from.u1(i1 false)
%xsum0.0 = call float @recipe.state.from.u1(i1 false)
%acc1.0 = call float @recipe.state.from.u1(i1 false)
%xsum1.0 = call float @recipe.state.from.u1(i1 false)
%acc2.0 = call float @recipe.state.from.u1(i1 false)
%xsum2.0 = call float @recipe.state.from.u1(i1 false)
%acc3.0 = call float @recipe.state.from.u1(i1 false)
%xsum3.0 = call float @recipe.state.from.u1(i1 false)
%ql0.shifted = lshr i32 %ql0.raw, %ql.shift
%ql0.code = and i32 %ql0.shifted, 15
%qh0.shifted = lshr i32 %qh0.raw, %qh.shift
%qh0.code = and i32 %qh0.shifted, 3
%qh0.bits = shl i32 %qh0.code, 4
%q0.joined = or i32 %ql0.code, %qh0.bits
%code0 = sub i32 %q0.joined, 32
%code0.state = call float @recipe.state.from.s32(i32 %code0)
%acc0.1 = call float @recipe.state.madd(float %acc0.0, float %code0.state, float %x0)
%xsum0.1 = call float @recipe.state.add(float %xsum0.0, float %x0)
%ql1.shifted = lshr i32 %ql1.raw, %ql.shift
%ql1.code = and i32 %ql1.shifted, 15
%qh1.shifted = lshr i32 %qh1.raw, %qh.shift
%qh1.code = and i32 %qh1.shifted, 3
%qh1.bits = shl i32 %qh1.code, 4
%q1.joined = or i32 %ql1.code, %qh1.bits
%code1 = sub i32 %q1.joined, 32
%code1.state = call float @recipe.state.from.s32(i32 %code1)
%acc1.1 = call float @recipe.state.madd(float %acc1.0, float %code1.state, float %x1)
%xsum1.1 = call float @recipe.state.add(float %xsum1.0, float %x1)
%ql2.shifted = lshr i32 %ql2.raw, %ql.shift
%ql2.code = and i32 %ql2.shifted, 15
%qh2.shifted = lshr i32 %qh2.raw, %qh.shift
%qh2.code = and i32 %qh2.shifted, 3
%qh2.bits = shl i32 %qh2.code, 4
%q2.joined = or i32 %ql2.code, %qh2.bits
%code2 = sub i32 %q2.joined, 32
%code2.state = call float @recipe.state.from.s32(i32 %code2)
%acc2.1 = call float @recipe.state.madd(float %acc2.0, float %code2.state, float %x2)
%xsum2.1 = call float @recipe.state.add(float %xsum2.0, float %x2)
%ql3.shifted = lshr i32 %ql3.raw, %ql.shift
%ql3.code = and i32 %ql3.shifted, 15
%qh3.shifted = lshr i32 %qh3.raw, %qh.shift
%qh3.code = and i32 %qh3.shifted, 3
%qh3.bits = shl i32 %qh3.code, 4
%q3.joined = or i32 %ql3.code, %qh3.bits
%code3 = sub i32 %q3.joined, 32
%code3.state = call float @recipe.state.from.s32(i32 %code3)
%acc3.1 = call float @recipe.state.madd(float %acc3.0, float %code3.state, float %x3)
%xsum3.1 = call float @recipe.state.add(float %xsum3.0, float %x3)
%ql4.shifted = lshr i32 %ql4.raw, %ql.shift
%ql4.code = and i32 %ql4.shifted, 15
%qh4.shifted = lshr i32 %qh4.raw, %qh.shift
%qh4.code = and i32 %qh4.shifted, 3
%qh4.bits = shl i32 %qh4.code, 4
%q4.joined = or i32 %ql4.code, %qh4.bits
%code4 = sub i32 %q4.joined, 32
%code4.state = call float @recipe.state.from.s32(i32 %code4)
%acc0.2 = call float @recipe.state.madd(float %acc0.1, float %code4.state, float %x4)
%xsum0.2 = call float @recipe.state.add(float %xsum0.1, float %x4)
%ql5.shifted = lshr i32 %ql5.raw, %ql.shift
%ql5.code = and i32 %ql5.shifted, 15
%qh5.shifted = lshr i32 %qh5.raw, %qh.shift
%qh5.code = and i32 %qh5.shifted, 3
%qh5.bits = shl i32 %qh5.code, 4
%q5.joined = or i32 %ql5.code, %qh5.bits
%code5 = sub i32 %q5.joined, 32
%code5.state = call float @recipe.state.from.s32(i32 %code5)
%acc1.2 = call float @recipe.state.madd(float %acc1.1, float %code5.state, float %x5)
%xsum1.2 = call float @recipe.state.add(float %xsum1.1, float %x5)
%ql6.shifted = lshr i32 %ql6.raw, %ql.shift
%ql6.code = and i32 %ql6.shifted, 15
%qh6.shifted = lshr i32 %qh6.raw, %qh.shift
%qh6.code = and i32 %qh6.shifted, 3
%qh6.bits = shl i32 %qh6.code, 4
%q6.joined = or i32 %ql6.code, %qh6.bits
%code6 = sub i32 %q6.joined, 32
%code6.state = call float @recipe.state.from.s32(i32 %code6)
%acc2.2 = call float @recipe.state.madd(float %acc2.1, float %code6.state, float %x6)
%xsum2.2 = call float @recipe.state.add(float %xsum2.1, float %x6)
%ql7.shifted = lshr i32 %ql7.raw, %ql.shift
%ql7.code = and i32 %ql7.shifted, 15
%qh7.shifted = lshr i32 %qh7.raw, %qh.shift
%qh7.code = and i32 %qh7.shifted, 3
%qh7.bits = shl i32 %qh7.code, 4
%q7.joined = or i32 %ql7.code, %qh7.bits
%code7 = sub i32 %q7.joined, 32
%code7.state = call float @recipe.state.from.s32(i32 %code7)
%acc3.2 = call float @recipe.state.madd(float %acc3.1, float %code7.state, float %x7)
%xsum3.2 = call float @recipe.state.add(float %xsum3.1, float %x7)
%ql8.shifted = lshr i32 %ql8.raw, %ql.shift
%ql8.code = and i32 %ql8.shifted, 15
%qh8.shifted = lshr i32 %qh8.raw, %qh.shift
%qh8.code = and i32 %qh8.shifted, 3
%qh8.bits = shl i32 %qh8.code, 4
%q8.joined = or i32 %ql8.code, %qh8.bits
%code8 = sub i32 %q8.joined, 32
%code8.state = call float @recipe.state.from.s32(i32 %code8)
%acc0.3 = call float @recipe.state.madd(float %acc0.2, float %code8.state, float %x8)
%xsum0.3 = call float @recipe.state.add(float %xsum0.2, float %x8)
%ql9.shifted = lshr i32 %ql9.raw, %ql.shift
%ql9.code = and i32 %ql9.shifted, 15
%qh9.shifted = lshr i32 %qh9.raw, %qh.shift
%qh9.code = and i32 %qh9.shifted, 3
%qh9.bits = shl i32 %qh9.code, 4
%q9.joined = or i32 %ql9.code, %qh9.bits
%code9 = sub i32 %q9.joined, 32
%code9.state = call float @recipe.state.from.s32(i32 %code9)
%acc1.3 = call float @recipe.state.madd(float %acc1.2, float %code9.state, float %x9)
%xsum1.3 = call float @recipe.state.add(float %xsum1.2, float %x9)
%ql10.shifted = lshr i32 %ql10.raw, %ql.shift
%ql10.code = and i32 %ql10.shifted, 15
%qh10.shifted = lshr i32 %qh10.raw, %qh.shift
%qh10.code = and i32 %qh10.shifted, 3
%qh10.bits = shl i32 %qh10.code, 4
%q10.joined = or i32 %ql10.code, %qh10.bits
%code10 = sub i32 %q10.joined, 32
%code10.state = call float @recipe.state.from.s32(i32 %code10)
%acc2.3 = call float @recipe.state.madd(float %acc2.2, float %code10.state, float %x10)
%xsum2.3 = call float @recipe.state.add(float %xsum2.2, float %x10)
%ql11.shifted = lshr i32 %ql11.raw, %ql.shift
%ql11.code = and i32 %ql11.shifted, 15
%qh11.shifted = lshr i32 %qh11.raw, %qh.shift
%qh11.code = and i32 %qh11.shifted, 3
%qh11.bits = shl i32 %qh11.code, 4
%q11.joined = or i32 %ql11.code, %qh11.bits
%code11 = sub i32 %q11.joined, 32
%code11.state = call float @recipe.state.from.s32(i32 %code11)
%acc3.3 = call float @recipe.state.madd(float %acc3.2, float %code11.state, float %x11)
%xsum3.3 = call float @recipe.state.add(float %xsum3.2, float %x11)
%ql12.shifted = lshr i32 %ql12.raw, %ql.shift
%ql12.code = and i32 %ql12.shifted, 15
%qh12.shifted = lshr i32 %qh12.raw, %qh.shift
%qh12.code = and i32 %qh12.shifted, 3
%qh12.bits = shl i32 %qh12.code, 4
%q12.joined = or i32 %ql12.code, %qh12.bits
%code12 = sub i32 %q12.joined, 32
%code12.state = call float @recipe.state.from.s32(i32 %code12)
%acc0.4 = call float @recipe.state.madd(float %acc0.3, float %code12.state, float %x12)
%xsum0.4 = call float @recipe.state.add(float %xsum0.3, float %x12)
%ql13.shifted = lshr i32 %ql13.raw, %ql.shift
%ql13.code = and i32 %ql13.shifted, 15
%qh13.shifted = lshr i32 %qh13.raw, %qh.shift
%qh13.code = and i32 %qh13.shifted, 3
%qh13.bits = shl i32 %qh13.code, 4
%q13.joined = or i32 %ql13.code, %qh13.bits
%code13 = sub i32 %q13.joined, 32
%code13.state = call float @recipe.state.from.s32(i32 %code13)
%acc1.4 = call float @recipe.state.madd(float %acc1.3, float %code13.state, float %x13)
%xsum1.4 = call float @recipe.state.add(float %xsum1.3, float %x13)
%ql14.shifted = lshr i32 %ql14.raw, %ql.shift
%ql14.code = and i32 %ql14.shifted, 15
%qh14.shifted = lshr i32 %qh14.raw, %qh.shift
%qh14.code = and i32 %qh14.shifted, 3
%qh14.bits = shl i32 %qh14.code, 4
%q14.joined = or i32 %ql14.code, %qh14.bits
%code14 = sub i32 %q14.joined, 32
%code14.state = call float @recipe.state.from.s32(i32 %code14)
%acc2.4 = call float @recipe.state.madd(float %acc2.3, float %code14.state, float %x14)
%xsum2.4 = call float @recipe.state.add(float %xsum2.3, float %x14)
%ql15.shifted = lshr i32 %ql15.raw, %ql.shift
%ql15.code = and i32 %ql15.shifted, 15
%qh15.shifted = lshr i32 %qh15.raw, %qh.shift
%qh15.code = and i32 %qh15.shifted, 3
%qh15.bits = shl i32 %qh15.code, 4
%q15.joined = or i32 %ql15.code, %qh15.bits
%code15 = sub i32 %q15.joined, 32
%code15.state = call float @recipe.state.from.s32(i32 %code15)
%acc3.4 = call float @recipe.state.madd(float %acc3.3, float %code15.state, float %x15)
%xsum3.4 = call float @recipe.state.add(float %xsum3.3, float %x15)
%acc01 = call float @recipe.state.add(float %acc0.4, float %acc1.4)
%acc23 = call float @recipe.state.add(float %acc2.4, float %acc3.4)
%acc16 = call float @recipe.state.add(float %acc01, float %acc23)
%xsum01 = call float @recipe.state.add(float %xsum0.4, float %xsum1.4)
%xsum23 = call float @recipe.state.add(float %xsum2.4, float %xsum3.4)
%xsum16 = call float @recipe.state.add(float %xsum01, float %xsum23)
%scaled = call float @recipe.state.mul(float %scale.state, float %acc16)
%result = call float @recipe.state.mul(float %d, float %scaled)
ret float %result
}
define internal float @recipe.block32.exact(i32 %kind, ptr addrspace(1) %weights, i64 %offset, ptr addrspace(3) %tile, i32 %column, i32 %pitch, i32 %slice) #1 {
entry:
%block = getelementptr i8, ptr addrspace(1) %weights, i64 %offset
%d.bits = load half, ptr addrspace(1) %block, align 2
%d = call float @recipe.state.from.f16(half %d.bits)
%is.iq4 = icmp eq i32 %kind, 1
%is.q40 = icmp eq i32 %kind, 2
%is.q41 = icmp eq i32 %kind, 3
%is.q80 = icmp eq i32 %kind, 4
%nibble.shift = mul i32 %slice, 4
%half.wide = zext i32 %slice to i64
%byte.half = mul i64 %half.wide, 16
%data.q41 = select i1 %is.q41, i64 4, i64 2
%data.base = select i1 %is.q80, i64 %byte.half, i64 0
%data = add i64 %data.q41, %data.base
%m.ptr = getelementptr i8, ptr addrspace(1) %block, i64 2
%m.bits = load half, ptr addrspace(1) %m.ptr, align 2
%m.loaded = call float @recipe.state.from.f16(half %m.bits)
%zero = call float @recipe.state.from.u1(i1 false)
%m = select i1 %is.q41, float %m.loaded, float %zero
%column.quad = mul i32 %column, 4
%xq0.row = mul i32 %pitch, 0
%xq0.slot = add i32 %xq0.row, %column.quad
%xq0.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq0.slot
%xq0 = load <4 x i8>, ptr addrspace(3) %xq0.ptr, align 4
%x0.model = extractelement <4 x i8> %xq0, i32 0
%x0 = call float @recipe.decode(i8 %x0.model)
%x1.model = extractelement <4 x i8> %xq0, i32 1
%x1 = call float @recipe.decode(i8 %x1.model)
%x2.model = extractelement <4 x i8> %xq0, i32 2
%x2 = call float @recipe.decode(i8 %x2.model)
%x3.model = extractelement <4 x i8> %xq0, i32 3
%x3 = call float @recipe.decode(i8 %x3.model)
%xq1.row = mul i32 %pitch, 1
%xq1.slot = add i32 %xq1.row, %column.quad
%xq1.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq1.slot
%xq1 = load <4 x i8>, ptr addrspace(3) %xq1.ptr, align 4
%x4.model = extractelement <4 x i8> %xq1, i32 0
%x4 = call float @recipe.decode(i8 %x4.model)
%x5.model = extractelement <4 x i8> %xq1, i32 1
%x5 = call float @recipe.decode(i8 %x5.model)
%x6.model = extractelement <4 x i8> %xq1, i32 2
%x6 = call float @recipe.decode(i8 %x6.model)
%x7.model = extractelement <4 x i8> %xq1, i32 3
%x7 = call float @recipe.decode(i8 %x7.model)
%xq2.row = mul i32 %pitch, 2
%xq2.slot = add i32 %xq2.row, %column.quad
%xq2.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq2.slot
%xq2 = load <4 x i8>, ptr addrspace(3) %xq2.ptr, align 4
%x8.model = extractelement <4 x i8> %xq2, i32 0
%x8 = call float @recipe.decode(i8 %x8.model)
%x9.model = extractelement <4 x i8> %xq2, i32 1
%x9 = call float @recipe.decode(i8 %x9.model)
%x10.model = extractelement <4 x i8> %xq2, i32 2
%x10 = call float @recipe.decode(i8 %x10.model)
%x11.model = extractelement <4 x i8> %xq2, i32 3
%x11 = call float @recipe.decode(i8 %x11.model)
%xq3.row = mul i32 %pitch, 3
%xq3.slot = add i32 %xq3.row, %column.quad
%xq3.ptr = getelementptr inbounds i8, ptr addrspace(3) %tile, i32 %xq3.slot
%xq3 = load <4 x i8>, ptr addrspace(3) %xq3.ptr, align 4
%x12.model = extractelement <4 x i8> %xq3, i32 0
%x12 = call float @recipe.decode(i8 %x12.model)
%x13.model = extractelement <4 x i8> %xq3, i32 1
%x13 = call float @recipe.decode(i8 %x13.model)
%x14.model = extractelement <4 x i8> %xq3, i32 2
%x14 = call float @recipe.decode(i8 %x14.model)
%x15.model = extractelement <4 x i8> %xq3, i32 3
%x15 = call float @recipe.decode(i8 %x15.model)
%bw.ptr = getelementptr i8, ptr addrspace(1) %block, i64 %data
%bw.addr = ptrtoint ptr addrspace(1) %bw.ptr to i64
%bw.low = and i64 %bw.addr, 15
%bw.line.addr = sub i64 %bw.addr, %bw.low
%bw.line = inttoptr i64 %bw.line.addr to ptr addrspace(1)
%bw.lo = load <4 x i32>, ptr addrspace(1) %bw.line, align 16
%bw.next = getelementptr i8, ptr addrspace(1) %bw.line, i64 16
%bw.hi = load <4 x i32>, ptr addrspace(1) %bw.next, align 16
%bw.q = lshr i64 %bw.low, 2
%bw.q32 = trunc i64 %bw.q to i32
%bw.r64 = and i64 %bw.low, 3
%bw.r32 = trunc i64 %bw.r64 to i32
%bw.r = shl i32 %bw.r32, 3
%bin0 = extractelement <4 x i32> %bw.lo, i32 0
%bin1 = extractelement <4 x i32> %bw.lo, i32 1
%bin2 = extractelement <4 x i32> %bw.lo, i32 2
%bin3 = extractelement <4 x i32> %bw.lo, i32 3
%bin4 = extractelement <4 x i32> %bw.hi, i32 0
%bin5 = extractelement <4 x i32> %bw.hi, i32 1
%bin6 = extractelement <4 x i32> %bw.hi, i32 2
%bin7 = extractelement <4 x i32> %bw.hi, i32 3
%bw.is0 = icmp eq i32 %bw.q32, 0
%bw.is1 = icmp eq i32 %bw.q32, 1
%bw.is2 = icmp eq i32 %bw.q32, 2
%bw.is3 = icmp eq i32 %bw.q32, 3
%bw0.a.23 = select i1 %bw.is2, i32 %bin2, i32 %bin3
%bw0.a.123 = select i1 %bw.is1, i32 %bin1, i32 %bw0.a.23
%bw0.a = select i1 %bw.is0, i32 %bin0, i32 %bw0.a.123
%bw0.b.23 = select i1 %bw.is2, i32 %bin3, i32 %bin4
%bw0.b.123 = select i1 %bw.is1, i32 %bin2, i32 %bw0.b.23
%bw0.b = select i1 %bw.is0, i32 %bin1, i32 %bw0.b.123
%bw0 = call i32 @llvm.fshr.i32(i32 %bw0.b, i32 %bw0.a, i32 %bw.r)
%bw1.a.23 = select i1 %bw.is2, i32 %bin3, i32 %bin4
%bw1.a.123 = select i1 %bw.is1, i32 %bin2, i32 %bw1.a.23
%bw1.a = select i1 %bw.is0, i32 %bin1, i32 %bw1.a.123
%bw1.b.23 = select i1 %bw.is2, i32 %bin4, i32 %bin5
%bw1.b.123 = select i1 %bw.is1, i32 %bin3, i32 %bw1.b.23
%bw1.b = select i1 %bw.is0, i32 %bin2, i32 %bw1.b.123
%bw1 = call i32 @llvm.fshr.i32(i32 %bw1.b, i32 %bw1.a, i32 %bw.r)
%bw2.a.23 = select i1 %bw.is2, i32 %bin4, i32 %bin5
%bw2.a.123 = select i1 %bw.is1, i32 %bin3, i32 %bw2.a.23
%bw2.a = select i1 %bw.is0, i32 %bin2, i32 %bw2.a.123
%bw2.b.23 = select i1 %bw.is2, i32 %bin5, i32 %bin6
%bw2.b.123 = select i1 %bw.is1, i32 %bin4, i32 %bw2.b.23
%bw2.b = select i1 %bw.is0, i32 %bin3, i32 %bw2.b.123
%bw2 = call i32 @llvm.fshr.i32(i32 %bw2.b, i32 %bw2.a, i32 %bw.r)
%bw3.a.23 = select i1 %bw.is2, i32 %bin5, i32 %bin6
%bw3.a.123 = select i1 %bw.is1, i32 %bin4, i32 %bw3.a.23
%bw3.a = select i1 %bw.is0, i32 %bin3, i32 %bw3.a.123
%bw3.b.23 = select i1 %bw.is2, i32 %bin6, i32 %bin7
%bw3.b.123 = select i1 %bw.is1, i32 %bin5, i32 %bw3.b.23
%bw3.b = select i1 %bw.is0, i32 %bin4, i32 %bw3.b.123
%bw3 = call i32 @llvm.fshr.i32(i32 %bw3.b, i32 %bw3.a, i32 %bw.r)
%b0.byte.shift = lshr i32 %bw0, 0
%b0.raw = and i32 %b0.byte.shift, 255
%b1.byte.shift = lshr i32 %bw0, 8
%b1.raw = and i32 %b1.byte.shift, 255
%b2.byte.shift = lshr i32 %bw0, 16
%b2.raw = and i32 %b2.byte.shift, 255
%b3.byte.shift = lshr i32 %bw0, 24
%b3.raw = and i32 %b3.byte.shift, 255
%b4.byte.shift = lshr i32 %bw1, 0
%b4.raw = and i32 %b4.byte.shift, 255
%b5.byte.shift = lshr i32 %bw1, 8
%b5.raw = and i32 %b5.byte.shift, 255
%b6.byte.shift = lshr i32 %bw1, 16
%b6.raw = and i32 %b6.byte.shift, 255
%b7.byte.shift = lshr i32 %bw1, 24
%b7.raw = and i32 %b7.byte.shift, 255
%b8.byte.shift = lshr i32 %bw2, 0
%b8.raw = and i32 %b8.byte.shift, 255
%b9.byte.shift = lshr i32 %bw2, 8
%b9.raw = and i32 %b9.byte.shift, 255
%b10.byte.shift = lshr i32 %bw2, 16
%b10.raw = and i32 %b10.byte.shift, 255
%b11.byte.shift = lshr i32 %bw2, 24
%b11.raw = and i32 %b11.byte.shift, 255
%b12.byte.shift = lshr i32 %bw3, 0
%b12.raw = and i32 %b12.byte.shift, 255
%b13.byte.shift = lshr i32 %bw3, 8
%b13.raw = and i32 %b13.byte.shift, 255
%b14.byte.shift = lshr i32 %bw3, 16
%b14.raw = and i32 %b14.byte.shift, 255
%b15.byte.shift = lshr i32 %bw3, 24
%b15.raw = and i32 %b15.byte.shift, 255
%acc0.0 = call float @recipe.state.from.u1(i1 false)
%xsum0.0 = call float @recipe.state.from.u1(i1 false)
%acc1.0 = call float @recipe.state.from.u1(i1 false)
%xsum1.0 = call float @recipe.state.from.u1(i1 false)
%acc2.0 = call float @recipe.state.from.u1(i1 false)
%xsum2.0 = call float @recipe.state.from.u1(i1 false)
%acc3.0 = call float @recipe.state.from.u1(i1 false)
%xsum3.0 = call float @recipe.state.from.u1(i1 false)
%b0.shifted = lshr i32 %b0.raw, %nibble.shift
%b0.nibble = and i32 %b0.shifted, 15
%b0.nibble.wide = zext i32 %b0.nibble to i64
%b0.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b0.nibble.wide
%b0.level.byte = load i8, ptr %b0.level.ptr, align 1
%b0.level = sext i8 %b0.level.byte to i32
%b0.byte = trunc i32 %b0.raw to i8
%b0.signed = sext i8 %b0.byte to i32
%b0.q40 = sub i32 %b0.nibble, 8
%b0.four = select i1 %is.q40, i32 %b0.q40, i32 %b0.nibble
%b0.nibble.or.level = select i1 %is.iq4, i32 %b0.level, i32 %b0.four
%code0 = select i1 %is.q80, i32 %b0.signed, i32 %b0.nibble.or.level
%code0.state = call float @recipe.state.from.s32(i32 %code0)
%acc0.1 = call float @recipe.state.madd(float %acc0.0, float %code0.state, float %x0)
%xsum0.1 = call float @recipe.state.add(float %xsum0.0, float %x0)
%b1.shifted = lshr i32 %b1.raw, %nibble.shift
%b1.nibble = and i32 %b1.shifted, 15
%b1.nibble.wide = zext i32 %b1.nibble to i64
%b1.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b1.nibble.wide
%b1.level.byte = load i8, ptr %b1.level.ptr, align 1
%b1.level = sext i8 %b1.level.byte to i32
%b1.byte = trunc i32 %b1.raw to i8
%b1.signed = sext i8 %b1.byte to i32
%b1.q40 = sub i32 %b1.nibble, 8
%b1.four = select i1 %is.q40, i32 %b1.q40, i32 %b1.nibble
%b1.nibble.or.level = select i1 %is.iq4, i32 %b1.level, i32 %b1.four
%code1 = select i1 %is.q80, i32 %b1.signed, i32 %b1.nibble.or.level
%code1.state = call float @recipe.state.from.s32(i32 %code1)
%acc1.1 = call float @recipe.state.madd(float %acc1.0, float %code1.state, float %x1)
%xsum1.1 = call float @recipe.state.add(float %xsum1.0, float %x1)
%b2.shifted = lshr i32 %b2.raw, %nibble.shift
%b2.nibble = and i32 %b2.shifted, 15
%b2.nibble.wide = zext i32 %b2.nibble to i64
%b2.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b2.nibble.wide
%b2.level.byte = load i8, ptr %b2.level.ptr, align 1
%b2.level = sext i8 %b2.level.byte to i32
%b2.byte = trunc i32 %b2.raw to i8
%b2.signed = sext i8 %b2.byte to i32
%b2.q40 = sub i32 %b2.nibble, 8
%b2.four = select i1 %is.q40, i32 %b2.q40, i32 %b2.nibble
%b2.nibble.or.level = select i1 %is.iq4, i32 %b2.level, i32 %b2.four
%code2 = select i1 %is.q80, i32 %b2.signed, i32 %b2.nibble.or.level
%code2.state = call float @recipe.state.from.s32(i32 %code2)
%acc2.1 = call float @recipe.state.madd(float %acc2.0, float %code2.state, float %x2)
%xsum2.1 = call float @recipe.state.add(float %xsum2.0, float %x2)
%b3.shifted = lshr i32 %b3.raw, %nibble.shift
%b3.nibble = and i32 %b3.shifted, 15
%b3.nibble.wide = zext i32 %b3.nibble to i64
%b3.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b3.nibble.wide
%b3.level.byte = load i8, ptr %b3.level.ptr, align 1
%b3.level = sext i8 %b3.level.byte to i32
%b3.byte = trunc i32 %b3.raw to i8
%b3.signed = sext i8 %b3.byte to i32
%b3.q40 = sub i32 %b3.nibble, 8
%b3.four = select i1 %is.q40, i32 %b3.q40, i32 %b3.nibble
%b3.nibble.or.level = select i1 %is.iq4, i32 %b3.level, i32 %b3.four
%code3 = select i1 %is.q80, i32 %b3.signed, i32 %b3.nibble.or.level
%code3.state = call float @recipe.state.from.s32(i32 %code3)
%acc3.1 = call float @recipe.state.madd(float %acc3.0, float %code3.state, float %x3)
%xsum3.1 = call float @recipe.state.add(float %xsum3.0, float %x3)
%b4.shifted = lshr i32 %b4.raw, %nibble.shift
%b4.nibble = and i32 %b4.shifted, 15
%b4.nibble.wide = zext i32 %b4.nibble to i64
%b4.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b4.nibble.wide
%b4.level.byte = load i8, ptr %b4.level.ptr, align 1
%b4.level = sext i8 %b4.level.byte to i32
%b4.byte = trunc i32 %b4.raw to i8
%b4.signed = sext i8 %b4.byte to i32
%b4.q40 = sub i32 %b4.nibble, 8
%b4.four = select i1 %is.q40, i32 %b4.q40, i32 %b4.nibble
%b4.nibble.or.level = select i1 %is.iq4, i32 %b4.level, i32 %b4.four
%code4 = select i1 %is.q80, i32 %b4.signed, i32 %b4.nibble.or.level
%code4.state = call float @recipe.state.from.s32(i32 %code4)
%acc0.2 = call float @recipe.state.madd(float %acc0.1, float %code4.state, float %x4)
%xsum0.2 = call float @recipe.state.add(float %xsum0.1, float %x4)
%b5.shifted = lshr i32 %b5.raw, %nibble.shift
%b5.nibble = and i32 %b5.shifted, 15
%b5.nibble.wide = zext i32 %b5.nibble to i64
%b5.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b5.nibble.wide
%b5.level.byte = load i8, ptr %b5.level.ptr, align 1
%b5.level = sext i8 %b5.level.byte to i32
%b5.byte = trunc i32 %b5.raw to i8
%b5.signed = sext i8 %b5.byte to i32
%b5.q40 = sub i32 %b5.nibble, 8
%b5.four = select i1 %is.q40, i32 %b5.q40, i32 %b5.nibble
%b5.nibble.or.level = select i1 %is.iq4, i32 %b5.level, i32 %b5.four
%code5 = select i1 %is.q80, i32 %b5.signed, i32 %b5.nibble.or.level
%code5.state = call float @recipe.state.from.s32(i32 %code5)
%acc1.2 = call float @recipe.state.madd(float %acc1.1, float %code5.state, float %x5)
%xsum1.2 = call float @recipe.state.add(float %xsum1.1, float %x5)
%b6.shifted = lshr i32 %b6.raw, %nibble.shift
%b6.nibble = and i32 %b6.shifted, 15
%b6.nibble.wide = zext i32 %b6.nibble to i64
%b6.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b6.nibble.wide
%b6.level.byte = load i8, ptr %b6.level.ptr, align 1
%b6.level = sext i8 %b6.level.byte to i32
%b6.byte = trunc i32 %b6.raw to i8
%b6.signed = sext i8 %b6.byte to i32
%b6.q40 = sub i32 %b6.nibble, 8
%b6.four = select i1 %is.q40, i32 %b6.q40, i32 %b6.nibble
%b6.nibble.or.level = select i1 %is.iq4, i32 %b6.level, i32 %b6.four
%code6 = select i1 %is.q80, i32 %b6.signed, i32 %b6.nibble.or.level
%code6.state = call float @recipe.state.from.s32(i32 %code6)
%acc2.2 = call float @recipe.state.madd(float %acc2.1, float %code6.state, float %x6)
%xsum2.2 = call float @recipe.state.add(float %xsum2.1, float %x6)
%b7.shifted = lshr i32 %b7.raw, %nibble.shift
%b7.nibble = and i32 %b7.shifted, 15
%b7.nibble.wide = zext i32 %b7.nibble to i64
%b7.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b7.nibble.wide
%b7.level.byte = load i8, ptr %b7.level.ptr, align 1
%b7.level = sext i8 %b7.level.byte to i32
%b7.byte = trunc i32 %b7.raw to i8
%b7.signed = sext i8 %b7.byte to i32
%b7.q40 = sub i32 %b7.nibble, 8
%b7.four = select i1 %is.q40, i32 %b7.q40, i32 %b7.nibble
%b7.nibble.or.level = select i1 %is.iq4, i32 %b7.level, i32 %b7.four
%code7 = select i1 %is.q80, i32 %b7.signed, i32 %b7.nibble.or.level
%code7.state = call float @recipe.state.from.s32(i32 %code7)
%acc3.2 = call float @recipe.state.madd(float %acc3.1, float %code7.state, float %x7)
%xsum3.2 = call float @recipe.state.add(float %xsum3.1, float %x7)
%b8.shifted = lshr i32 %b8.raw, %nibble.shift
%b8.nibble = and i32 %b8.shifted, 15
%b8.nibble.wide = zext i32 %b8.nibble to i64
%b8.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b8.nibble.wide
%b8.level.byte = load i8, ptr %b8.level.ptr, align 1
%b8.level = sext i8 %b8.level.byte to i32
%b8.byte = trunc i32 %b8.raw to i8
%b8.signed = sext i8 %b8.byte to i32
%b8.q40 = sub i32 %b8.nibble, 8
%b8.four = select i1 %is.q40, i32 %b8.q40, i32 %b8.nibble
%b8.nibble.or.level = select i1 %is.iq4, i32 %b8.level, i32 %b8.four
%code8 = select i1 %is.q80, i32 %b8.signed, i32 %b8.nibble.or.level
%code8.state = call float @recipe.state.from.s32(i32 %code8)
%acc0.3 = call float @recipe.state.madd(float %acc0.2, float %code8.state, float %x8)
%xsum0.3 = call float @recipe.state.add(float %xsum0.2, float %x8)
%b9.shifted = lshr i32 %b9.raw, %nibble.shift
%b9.nibble = and i32 %b9.shifted, 15
%b9.nibble.wide = zext i32 %b9.nibble to i64
%b9.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b9.nibble.wide
%b9.level.byte = load i8, ptr %b9.level.ptr, align 1
%b9.level = sext i8 %b9.level.byte to i32
%b9.byte = trunc i32 %b9.raw to i8
%b9.signed = sext i8 %b9.byte to i32
%b9.q40 = sub i32 %b9.nibble, 8
%b9.four = select i1 %is.q40, i32 %b9.q40, i32 %b9.nibble
%b9.nibble.or.level = select i1 %is.iq4, i32 %b9.level, i32 %b9.four
%code9 = select i1 %is.q80, i32 %b9.signed, i32 %b9.nibble.or.level
%code9.state = call float @recipe.state.from.s32(i32 %code9)
%acc1.3 = call float @recipe.state.madd(float %acc1.2, float %code9.state, float %x9)
%xsum1.3 = call float @recipe.state.add(float %xsum1.2, float %x9)
%b10.shifted = lshr i32 %b10.raw, %nibble.shift
%b10.nibble = and i32 %b10.shifted, 15
%b10.nibble.wide = zext i32 %b10.nibble to i64
%b10.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b10.nibble.wide
%b10.level.byte = load i8, ptr %b10.level.ptr, align 1
%b10.level = sext i8 %b10.level.byte to i32
%b10.byte = trunc i32 %b10.raw to i8
%b10.signed = sext i8 %b10.byte to i32
%b10.q40 = sub i32 %b10.nibble, 8
%b10.four = select i1 %is.q40, i32 %b10.q40, i32 %b10.nibble
%b10.nibble.or.level = select i1 %is.iq4, i32 %b10.level, i32 %b10.four
%code10 = select i1 %is.q80, i32 %b10.signed, i32 %b10.nibble.or.level
%code10.state = call float @recipe.state.from.s32(i32 %code10)
%acc2.3 = call float @recipe.state.madd(float %acc2.2, float %code10.state, float %x10)
%xsum2.3 = call float @recipe.state.add(float %xsum2.2, float %x10)
%b11.shifted = lshr i32 %b11.raw, %nibble.shift
%b11.nibble = and i32 %b11.shifted, 15
%b11.nibble.wide = zext i32 %b11.nibble to i64
%b11.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b11.nibble.wide
%b11.level.byte = load i8, ptr %b11.level.ptr, align 1
%b11.level = sext i8 %b11.level.byte to i32
%b11.byte = trunc i32 %b11.raw to i8
%b11.signed = sext i8 %b11.byte to i32
%b11.q40 = sub i32 %b11.nibble, 8
%b11.four = select i1 %is.q40, i32 %b11.q40, i32 %b11.nibble
%b11.nibble.or.level = select i1 %is.iq4, i32 %b11.level, i32 %b11.four
%code11 = select i1 %is.q80, i32 %b11.signed, i32 %b11.nibble.or.level
%code11.state = call float @recipe.state.from.s32(i32 %code11)
%acc3.3 = call float @recipe.state.madd(float %acc3.2, float %code11.state, float %x11)
%xsum3.3 = call float @recipe.state.add(float %xsum3.2, float %x11)
%b12.shifted = lshr i32 %b12.raw, %nibble.shift
%b12.nibble = and i32 %b12.shifted, 15
%b12.nibble.wide = zext i32 %b12.nibble to i64
%b12.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b12.nibble.wide
%b12.level.byte = load i8, ptr %b12.level.ptr, align 1
%b12.level = sext i8 %b12.level.byte to i32
%b12.byte = trunc i32 %b12.raw to i8
%b12.signed = sext i8 %b12.byte to i32
%b12.q40 = sub i32 %b12.nibble, 8
%b12.four = select i1 %is.q40, i32 %b12.q40, i32 %b12.nibble
%b12.nibble.or.level = select i1 %is.iq4, i32 %b12.level, i32 %b12.four
%code12 = select i1 %is.q80, i32 %b12.signed, i32 %b12.nibble.or.level
%code12.state = call float @recipe.state.from.s32(i32 %code12)
%acc0.4 = call float @recipe.state.madd(float %acc0.3, float %code12.state, float %x12)
%xsum0.4 = call float @recipe.state.add(float %xsum0.3, float %x12)
%b13.shifted = lshr i32 %b13.raw, %nibble.shift
%b13.nibble = and i32 %b13.shifted, 15
%b13.nibble.wide = zext i32 %b13.nibble to i64
%b13.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b13.nibble.wide
%b13.level.byte = load i8, ptr %b13.level.ptr, align 1
%b13.level = sext i8 %b13.level.byte to i32
%b13.byte = trunc i32 %b13.raw to i8
%b13.signed = sext i8 %b13.byte to i32
%b13.q40 = sub i32 %b13.nibble, 8
%b13.four = select i1 %is.q40, i32 %b13.q40, i32 %b13.nibble
%b13.nibble.or.level = select i1 %is.iq4, i32 %b13.level, i32 %b13.four
%code13 = select i1 %is.q80, i32 %b13.signed, i32 %b13.nibble.or.level
%code13.state = call float @recipe.state.from.s32(i32 %code13)
%acc1.4 = call float @recipe.state.madd(float %acc1.3, float %code13.state, float %x13)
%xsum1.4 = call float @recipe.state.add(float %xsum1.3, float %x13)
%b14.shifted = lshr i32 %b14.raw, %nibble.shift
%b14.nibble = and i32 %b14.shifted, 15
%b14.nibble.wide = zext i32 %b14.nibble to i64
%b14.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b14.nibble.wide
%b14.level.byte = load i8, ptr %b14.level.ptr, align 1
%b14.level = sext i8 %b14.level.byte to i32
%b14.byte = trunc i32 %b14.raw to i8
%b14.signed = sext i8 %b14.byte to i32
%b14.q40 = sub i32 %b14.nibble, 8
%b14.four = select i1 %is.q40, i32 %b14.q40, i32 %b14.nibble
%b14.nibble.or.level = select i1 %is.iq4, i32 %b14.level, i32 %b14.four
%code14 = select i1 %is.q80, i32 %b14.signed, i32 %b14.nibble.or.level
%code14.state = call float @recipe.state.from.s32(i32 %code14)
%acc2.4 = call float @recipe.state.madd(float %acc2.3, float %code14.state, float %x14)
%xsum2.4 = call float @recipe.state.add(float %xsum2.3, float %x14)
%b15.shifted = lshr i32 %b15.raw, %nibble.shift
%b15.nibble = and i32 %b15.shifted, 15
%b15.nibble.wide = zext i32 %b15.nibble to i64
%b15.level.ptr = getelementptr inbounds [16 x i8], ptr @recipe_block32_iq4_levels, i64 0, i64 %b15.nibble.wide
%b15.level.byte = load i8, ptr %b15.level.ptr, align 1
%b15.level = sext i8 %b15.level.byte to i32
%b15.byte = trunc i32 %b15.raw to i8
%b15.signed = sext i8 %b15.byte to i32
%b15.q40 = sub i32 %b15.nibble, 8
%b15.four = select i1 %is.q40, i32 %b15.q40, i32 %b15.nibble
%b15.nibble.or.level = select i1 %is.iq4, i32 %b15.level, i32 %b15.four
%code15 = select i1 %is.q80, i32 %b15.signed, i32 %b15.nibble.or.level
%code15.state = call float @recipe.state.from.s32(i32 %code15)
%acc3.4 = call float @recipe.state.madd(float %acc3.3, float %code15.state, float %x15)
%xsum3.4 = call float @recipe.state.add(float %xsum3.3, float %x15)
%acc01 = call float @recipe.state.add(float %acc0.4, float %acc1.4)
%acc23 = call float @recipe.state.add(float %acc2.4, float %acc3.4)
%acc16 = call float @recipe.state.add(float %acc01, float %acc23)
%xsum01 = call float @recipe.state.add(float %xsum0.4, float %xsum1.4)
%xsum23 = call float @recipe.state.add(float %xsum2.4, float %xsum3.4)
%xsum16 = call float @recipe.state.add(float %xsum01, float %xsum23)
%dot.scaled = call float @recipe.state.mul(float %d, float %acc16)
%minimum.scaled = call float @recipe.state.mul(float %m, float %xsum16)
%result = call float @recipe.state.add(float %dot.scaled, float %minimum.scaled)
ret float %result
}

declare void @llvm.trap() @contraction_tile_f8 = external addrspace(3) global [0 x i8], align 16
define internal i8 @contraction_input(
ptr addrspace(1) %input, i64 %row.base, i32 %position, i32 %term, i32 %span, i32 %length, i1 %conv ) #1 { entry:
%channel = udiv i32 %term, %span %window = urem i32 %term, %span
%offset = select i1 %conv, i32 %window, i32 0 %channel.wide = zext i32 %channel to i64 %length.wide = zext i32 %length to i64 %position.wide = zext i32 %position to i64 %offset.wide = zext i32 %offset to i64
%channel.base = mul i64 %channel.wide, %length.wide
%local.0 = add i64 %channel.base, %position.wide %local = add i64 %local.0, %offset.wide
%index = add i64 %row.base, %local %ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %index
%value = load i8, ptr addrspace(1) %ptr, align 1 ret i8 %value }

; The backward passes read the delta in the arithmetic type: nothing on the adjoint
; side is ever rounded to the model type. The relu gate still reads the model output.
define internal float @contraction_delta_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %index
%delta.value = load float, ptr addrspace(1) %delta.ptr, align 4
%zero = call float @recipe.state.from.u1(i1 false)
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %index
%output.value = load i8, ptr addrspace(1) %output.ptr, align 1
%positive = call i1 @recipe.ogt(i8 %output.value, i8 0)
%activated = select i1 %positive, float %delta.value, float %zero
br label %done
done:
%value = phi float [ %delta.value, %entry ], [ %activated, %activation ]
ret float %value
}
define internal <16 x float> @contraction_delta_vector16_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %index
%delta.value = load <16 x float>, ptr addrspace(1) %delta.ptr, align 4
%zero = call float @recipe.state.from.u1(i1 false)
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %index
%output.value = load <16 x i8>, ptr addrspace(1) %output.ptr, align 1
br label %activation.loop
activation.loop:
%activation.index = phi i32 [ 0, %activation ], [ %activation.next, %activation.step ]
%activation.values = phi <16 x float> [ zeroinitializer, %activation ], [ %activation.values.next, %activation.step ]
%activation.more = icmp ult i32 %activation.index, 16
br i1 %activation.more, label %activation.step, label %done
activation.step:
%activation.output = extractelement <16 x i8> %output.value, i32 %activation.index
%activation.delta = extractelement <16 x float> %delta.value, i32 %activation.index
%activation.positive = call i1 @recipe.ogt(i8 %activation.output, i8 0)
%activation.value = select i1 %activation.positive, float %activation.delta, float %zero
%activation.values.next = insertelement <16 x float> %activation.values, float %activation.value, i32 %activation.index
%activation.next = add i32 %activation.index, 1
br label %activation.loop
done:
%value = phi <16 x float> [ %delta.value, %entry ], [ %activation.values, %activation.loop ]
ret <16 x float> %value
}
define internal void @reduce_rows_state(ptr addrspace(1) %source, ptr addrspace(1) %target, i32 %rows, i32 %columns, i32 %stride, i32 %source.offset, i32 %target.offset, i32 %threads) #1 {
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
%sum.next = call float @recipe.state.add(float %sum, float %source.value)
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
define internal i32 @contraction_matrix_a_index(i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
%row = mul i32 %m, %tile.k
%index = add i32 %row, %k
ret i32 %index
}
define internal i32 @contraction_vector_b_index(i32 %k, i32 %n, i32 %tile.n, i32 %tile.k) #1 {
entry:
%row = mul i32 %k, %tile.n
%index = add i32 %row, %n
ret i32 %index
}
define internal i32 @contraction_matrix_b_index(i32 %k, i32 %n, i32 %tile.n, i32 %tile.k) #1 {
entry:
%row = mul i32 %n, %tile.k
%index = add i32 %row, %k
ret i32 %index
}
define internal void @contraction_stage_column16(<16 x i8> %values, i32 %base, i32 %r, i32 %column, i32 %stride) #1 {
entry:
br label %loop
loop:
%j = phi i32 [ 0, %entry ], [ %j.next, %step ]
%more = icmp ult i32 %j, 16
br i1 %more, label %step, label %done
step:
%value = extractelement <16 x i8> %values, i32 %j
%row.local = add i32 %r, %j
%row = mul i32 %row.local, %stride
%local = add i32 %row, %column
%index = add i32 %base, %local
%ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 %value, ptr addrspace(3) %ptr, align 1
%j.next = add i32 %j, 1
br label %loop
done:
ret void
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
%ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 0, ptr addrspace(3) %ptr, align 1
%next = add i32 %p, %block
br label %loop
exit:
ret void
}
define internal void @contraction_zero_edges_bs(i32 %m.count, i32 %n.count, i32 %k.count, i32 %lid, i32 %block, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
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
br label %store.a
b.step:
%b.p = sub i32 %p, %a.count
%b.k = udiv i32 %b.p, %b.missing
%b.local = urem i32 %b.p, %b.missing
%b.n = add i32 %n.count, %b.local
%b.elements = mul i32 %tile.m, %tile.k
%b.base = call i32 @contraction_state_after_model(i32 %b.elements)
%b.local.index = call i32 @contraction_vector_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k)
%b.index = add i32 %b.base, %b.local.index
br label %store.b
a.k.step:
%a.k.p = sub i32 %p, %output.count
%a.k.local = udiv i32 %a.k.p, %m.count
%a.k.value = add i32 %k.count, %a.k.local
%a.k.m = urem i32 %a.k.p, %m.count
%a.k.index = call i32 @contraction_vector_a_index(i32 %a.k.value, i32 %a.k.m, i32 %tile.m, i32 %tile.k)
br label %store.a
b.k.step:
%b.k.p = sub i32 %p, %a.k.limit
%b.k.local = udiv i32 %b.k.p, %n.count
%b.k.value = add i32 %k.count, %b.k.local
%b.k.n = urem i32 %b.k.p, %n.count
%b.k.elements = mul i32 %tile.m, %tile.k
%b.k.base = call i32 @contraction_state_after_model(i32 %b.k.elements)
%b.k.local.index = call i32 @contraction_vector_b_index(i32 %b.k.value, i32 %b.k.n, i32 %tile.n, i32 %tile.k)
%b.k.index = add i32 %b.k.base, %b.k.local.index
br label %store.b
store.a:
%a.any = phi i32 [ %a.index, %a.step ], [ %a.k.index, %a.k.step ]
%a.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %a.any
store i8 0, ptr addrspace(3) %a.ptr, align 1
br label %store
store.b:
%b.any = phi i32 [ %b.index, %b.step ], [ %b.k.index, %b.k.step ]
%b.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %b.any
store float %state.zero, ptr addrspace(3) %b.ptr, align 4
br label %store
store:
%next = add i32 %p, %block
br label %loop
exit:
ret void
}
define internal void @contraction_zero_edges_as(i32 %m.count, i32 %n.count, i32 %k.count, i32 %lid, i32 %block, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
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
br label %store.a
b.step:
%b.p = sub i32 %p, %a.count
%b.k = udiv i32 %b.p, %b.missing
%b.local = urem i32 %b.p, %b.missing
%b.n = add i32 %n.count, %b.local
%b.elements = mul i32 %tile.m, %tile.k
%b.base = call i32 @contraction_model_after_state(i32 %b.elements)
%b.local.index = call i32 @contraction_vector_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k)
%b.index = add i32 %b.base, %b.local.index
br label %store.b
a.k.step:
%a.k.p = sub i32 %p, %output.count
%a.k.local = udiv i32 %a.k.p, %m.count
%a.k.value = add i32 %k.count, %a.k.local
%a.k.m = urem i32 %a.k.p, %m.count
%a.k.index = call i32 @contraction_vector_a_index(i32 %a.k.value, i32 %a.k.m, i32 %tile.m, i32 %tile.k)
br label %store.a
b.k.step:
%b.k.p = sub i32 %p, %a.k.limit
%b.k.local = udiv i32 %b.k.p, %n.count
%b.k.value = add i32 %k.count, %b.k.local
%b.k.n = urem i32 %b.k.p, %n.count
%b.k.elements = mul i32 %tile.m, %tile.k
%b.k.base = call i32 @contraction_model_after_state(i32 %b.k.elements)
%b.k.local.index = call i32 @contraction_vector_b_index(i32 %b.k.value, i32 %b.k.n, i32 %tile.n, i32 %tile.k)
%b.k.index = add i32 %b.k.base, %b.k.local.index
br label %store.b
store.a:
%a.any = phi i32 [ %a.index, %a.step ], [ %a.k.index, %a.k.step ]
%a.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %a.any
store float %state.zero, ptr addrspace(3) %a.ptr, align 4
br label %store
store.b:
%b.any = phi i32 [ %b.index, %b.step ], [ %b.k.index, %b.k.step ]
%b.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %b.any
store i8 0, ptr addrspace(3) %b.ptr, align 1
br label %store
store:
%next = add i32 %p, %block
br label %loop
exit:
ret void
}
define internal <RECIPE_REGISTER_M x i8> @contraction_a_fragment(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k) #1 {
entry:
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_M x i8>, ptr addrspace(3) %ptr, align 1
ret <RECIPE_REGISTER_M x i8> %fragment
}
define internal <RECIPE_REGISTER_N x i8> @contraction_b_fragment(i32 %k, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %output.n.base, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_N x i8>, ptr addrspace(3) %ptr, align 1
ret <RECIPE_REGISTER_N x i8> %fragment
}
define internal void @contraction_stage_a_fragment(<RECIPE_FRAGMENT_K x i8> %fragment, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.k = add i32 %k, %element
%index = call i32 @contraction_vector_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
%value = extractelement <RECIPE_FRAGMENT_K x i8> %fragment, i32 %element
%target = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 %value, ptr addrspace(3) %target, align 1
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_a_columns(<RECIPE_FRAGMENT_K x i8> %fragment, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.m = add i32 %m, %element
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %local.m, i32 %tile.m, i32 %tile.k)
%value = extractelement <RECIPE_FRAGMENT_K x i8> %fragment, i32 %element
%target = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 %value, ptr addrspace(3) %target, align 1
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
; Stage a fragment of consecutive K for one channel. The B tile is k-major, so
; consecutive K lands one %tile.n row apart and the elements are placed through
; @contraction_b_index rather than stored as one contiguous vector.
define internal void @contraction_stage_b_terms(<RECIPE_FRAGMENT_K x i8> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.k = add i32 %k, %element
%local = call i32 @contraction_vector_b_index(i32 %local.k, i32 %n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <RECIPE_FRAGMENT_K x i8> %fragment, i32 %element
%target = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 %value, ptr addrspace(3) %target, align 1
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_b_fragment(<RECIPE_FRAGMENT_K x i8> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.n = add i32 %n, %element
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %local.n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <RECIPE_FRAGMENT_K x i8> %fragment, i32 %element
%target = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 %value, ptr addrspace(3) %target, align 1
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_delta_a_fragment(<RECIPE_FRAGMENT_K x i8> %delta, <RECIPE_FRAGMENT_K x i8> %output, i1 %relu, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%delta.value = extractelement <RECIPE_FRAGMENT_K x i8> %delta, i32 %element
%output.value = extractelement <RECIPE_FRAGMENT_K x i8> %output, i32 %element
%positive = call i1 @recipe.ogt(i8 %output.value, i8 0)
%active = select i1 %positive, i8 %delta.value, i8 0
%value = select i1 %relu, i8 %active, i8 %delta.value
%local.k = add i32 %k, %element
%index = call i32 @contraction_vector_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
%target = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 %value, ptr addrspace(3) %target, align 1
%next = add i32 %element, 1
br label %loop
exit:
ret void
}

; Backward tiles hold the adjoint operand in float. The gradient pass keeps
; A (the input) in the model type at its usual index and puts B (the delta) in a
; state region that starts after the model A region; the previous pass puts A (the
; delta) in a state region at the tile's start and B (the weights) in the model
; type after it. Each base is whole elements of its region's own type, and the
; backward passes always take the vector layout.
define internal i32 @contraction_state_after_model(i32 %elements) #1 {
entry:
%bytes = mul i32 %elements, 1
%pad = sub i32 4, 1
%rounded = add i32 %bytes, %pad
%base = udiv i32 %rounded, 4
ret i32 %base
}
define internal i32 @contraction_model_after_state(i32 %elements) #1 {
entry:
%bytes = mul i32 %elements, 4
%pad = sub i32 1, 1
%rounded = add i32 %bytes, %pad
%base = udiv i32 %rounded, 1
ret i32 %base
}
define internal <RECIPE_REGISTER_M x i8> @contraction_a_fragment_vector(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k) #1 {
entry:
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_M x i8>, ptr addrspace(3) %ptr, align 1
ret <RECIPE_REGISTER_M x i8> %fragment
}
define internal <RECIPE_REGISTER_M x float> @contraction_a_fragment_state(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k) #1 {
entry:
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %index
%fragment = load <RECIPE_REGISTER_M x float>, ptr addrspace(3) %ptr, align 4
ret <RECIPE_REGISTER_M x float> %fragment
}
define internal <RECIPE_REGISTER_N x float> @contraction_b_fragment_state(i32 %k, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%a.elements = mul i32 %tile.m, %tile.k
%base = call i32 @contraction_state_after_model(i32 %a.elements)
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %output.n.base, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %index
%fragment = load <RECIPE_REGISTER_N x float>, ptr addrspace(3) %ptr, align 4
ret <RECIPE_REGISTER_N x float> %fragment
}
define internal <RECIPE_REGISTER_N x i8> @contraction_b_fragment_after_state(i32 %k, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%a.elements = mul i32 %tile.m, %tile.k
%base = call i32 @contraction_model_after_state(i32 %a.elements)
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %output.n.base, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_N x i8>, ptr addrspace(3) %ptr, align 1
ret <RECIPE_REGISTER_N x i8> %fragment
}
define internal void @contraction_stage_b_fragment_state(<RECIPE_FRAGMENT_K x float> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%a.elements = mul i32 %tile.m, %tile.k
%base = call i32 @contraction_state_after_model(i32 %a.elements)
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.n = add i32 %n, %element
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %local.n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <RECIPE_FRAGMENT_K x float> %fragment, i32 %element
%target = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %index
store float %value, ptr addrspace(3) %target, align 4
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_b_fragment_after_state(<RECIPE_FRAGMENT_K x i8> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%a.elements = mul i32 %tile.m, %tile.k
%base = call i32 @contraction_model_after_state(i32 %a.elements)
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.n = add i32 %n, %element
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %local.n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <RECIPE_FRAGMENT_K x i8> %fragment, i32 %element
%target = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
store i8 %value, ptr addrspace(3) %target, align 1
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_delta_a_fragment_state(<RECIPE_FRAGMENT_K x float> %delta, <RECIPE_FRAGMENT_K x i8> %output, i1 %relu, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
%zero = call float @recipe.state.from.u1(i1 false)
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%delta.value = extractelement <RECIPE_FRAGMENT_K x float> %delta, i32 %element
%output.value = extractelement <RECIPE_FRAGMENT_K x i8> %output, i32 %element
%positive = call i1 @recipe.ogt(i8 %output.value, i8 0)
%active = select i1 %positive, float %delta.value, float %zero
%value = select i1 %relu, float %active, float %delta.value
%local.k = add i32 %k, %element
%index = call i32 @contraction_vector_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
%target = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %index
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
%base = mul i32 %lane, RECIPE_REGISTER_M
%local = urem i32 %register, RECIPE_REGISTER_M
%m = add i32 %base, %local
ret i32 %m
}
define internal i32 @contraction_vector_output_n(i32 %lid, i32 %register, i32 %m.lanes) #1 {
entry:
%lane = udiv i32 %lid, %m.lanes
%base = mul i32 %lane, RECIPE_REGISTER_N
%local = udiv i32 %register, RECIPE_REGISTER_M
%n = add i32 %base, %local
ret i32 %n
}
define internal i32 @contraction_matrix_output_m(i32 %lid, i32 %register, i32 %m.lanes) #1 {
entry:
%wave = udiv i32 %lid, 32
%lane = urem i32 %lid, 32
%half = udiv i32 %lane, 16
%wave.base = mul i32 %wave, 16
%local.twice = mul i32 %register, 2
%local = urem i32 %local.twice, 16
%row = add i32 %local, %half
%m = add i32 %wave.base, %row
ret i32 %m
}
define internal i32 @contraction_matrix_output_n(i32 %lid, i32 %register, i32 %m.lanes) #1 {
entry:
%lane = urem i32 %lid, 16
%tile = udiv i32 %register, 8
%base = mul i32 %tile, 16
%n = add i32 %base, %lane
ret i32 %n
}
define internal i1 @contraction_vector_store_lane(i1 %store, i32 %lid) #1 {
entry:
ret i1 %store
}
define internal i1 @contraction_matrix_store_lane(i1 %store, i32 %lid) #1 {
entry:
ret i1 true
}
define internal i1 @contraction_output_register_valid(i32 %register) #1 {
entry:
ret i1 true
}
; The K walk is cut into fixed chunks of RECIPE_CHUNK_K elements. Each chunk
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
%chunk.sums = alloca [RECIPE_CHUNK_VALUES x float], align 4
%state.zero = call float @recipe.state.from.u1(i1 false)
%chunks.adjusted = add i32 %k.count, RECIPE_CHUNK_K
%chunks.numerator = sub i32 %chunks.adjusted, 1
%chunks = udiv i32 %chunks.numerator, RECIPE_CHUNK_K
%single.sum.slot = icmp eq i32 RECIPE_CHUNK_VALUES, RECIPE_REGISTER_COUNT
%publish.lane.width = mul i32 %output.lanes, RECIPE_REGISTER_COUNT
%chunk.first = select i1 %lane.active, i32 %lane.k, i32 %chunks
%local.lane = icmp eq i32 %k.lanes, 1
%few.chunks = icmp ule i32 %chunks, 3
%local = or i1 %local.lane, %few.chunks
%local.owner = icmp eq i32 %lane.k, 0
%local.owner.active = and i1 %lane.active, %local.owner
br i1 %local, label %local.owner.check, label %shared.chunk.loop
local.owner.check:
br i1 %local.owner.active, label %local.k.begin, label %exit
; Use the same chunk boundaries and fold as the shared path, even when
; one lane computes every chunk. The schedule must not change rounding.
local.k.begin:
br label %local.chunk.loop
local.chunk.loop:
%local.chunk.first = phi i32 [ 0, %local.k.begin ], [ %local.chunk.limit, %local.store ]
%local.chunk.end = add i32 %local.chunk.first, RECIPE_CHUNK_K
%local.chunk.over = icmp ugt i32 %local.chunk.end, %k.count
%local.chunk.limit = select i1 %local.chunk.over, i32 %k.count, i32 %local.chunk.end
%local.sums.prior = load <RECIPE_REGISTER_COUNT x float>, ptr %sums, align 4
%local.a.initial = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment(i32 %local.chunk.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.initial = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment(i32 %local.chunk.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %local.k.loop
local.k.loop:
%local.k = phi i32 [ %local.chunk.first, %local.chunk.loop ], [ %local.k.next, %local.product.done ]
%local.sums = phi <RECIPE_REGISTER_COUNT x float> [ zeroinitializer, %local.chunk.loop ], [ %local.sums.current, %local.product.done ]
%local.a.fragment = phi <RECIPE_REGISTER_M x i8> [ %local.a.initial, %local.chunk.loop ], [ %local.a.next, %local.product.done ]
%local.b.fragment = phi <RECIPE_REGISTER_N x i8> [ %local.b.initial, %local.chunk.loop ], [ %local.b.next, %local.product.done ]
%local.k.next = add i32 %local.k, 1
%local.k.more = icmp ult i32 %local.k.next, %local.chunk.limit
%local.k.prefetch = select i1 %local.k.more, i32 %local.k.next, i32 %local.k
%local.a.next = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment(i32 %local.k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.next = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment(i32 %local.k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%local.a.wide = call <RECIPE_REGISTER_M x float> @contraction_widen_m(<RECIPE_REGISTER_M x i8> %local.a.fragment)
br label %local.product.loop
local.product.loop:
%local.product = phi i32 [ 0, %local.k.loop ], [ %local.product.next, %local.product.step ]
%local.sums.current = phi <RECIPE_REGISTER_COUNT x float> [ %local.sums, %local.k.loop ], [ %local.candidate, %local.product.step ]
%local.product.more = icmp ult i32 %local.product, RECIPE_REGISTER_COUNT
br i1 %local.product.more, label %local.product.step, label %local.product.done
local.product.step:
%local.a.index = urem i32 %local.product, RECIPE_REGISTER_M
%local.b.index = udiv i32 %local.product, RECIPE_REGISTER_M
%local.a = extractelement <RECIPE_REGISTER_M x float> %local.a.wide, i32 %local.a.index
%local.b = extractelement <RECIPE_REGISTER_N x i8> %local.b.fragment, i32 %local.b.index
%local.b.wide = call float @recipe.decode(i8 %local.b)
%local.sum = extractelement <RECIPE_REGISTER_COUNT x float> %local.sums.current, i32 %local.product
%local.value = call float @recipe.state.madd(float %local.sum, float %local.a, float %local.b.wide)
%local.candidate = insertelement <RECIPE_REGISTER_COUNT x float> %local.sums.current, float %local.value, i32 %local.product
%local.product.next = add i32 %local.product, 1
br label %local.product.loop, !llvm.loop !0
local.product.done:
br i1 %local.k.more, label %local.k.loop, label %local.store
local.store:
%local.folded = fadd <RECIPE_REGISTER_COUNT x float> %local.sums.prior, %local.sums.current
store <RECIPE_REGISTER_COUNT x float> %local.folded, ptr %sums, align 4
%local.chunk.more = icmp ult i32 %local.chunk.limit, %k.count
br i1 %local.chunk.more, label %local.chunk.loop, label %exit
shared.chunk.loop:
%chunk = phi i32 [ %chunk.first, %entry ], [ %chunk.next, %chunk.finish ]
%slot = phi i32 [ 0, %entry ], [ %slot.next, %chunk.finish ]
%sum.slot = select i1 %single.sum.slot, i32 0, i32 %slot
%chunk.more = icmp ult i32 %chunk, %chunks
br i1 %chunk.more, label %chunk.zero.loop, label %chunk.done
chunk.zero.loop:
%zero.r = phi i32 [ 0, %shared.chunk.loop ], [ %zero.next, %chunk.zero.step ]
%zero.more = icmp ult i32 %zero.r, RECIPE_REGISTER_COUNT
br i1 %zero.more, label %chunk.zero.step, label %k.begin
chunk.zero.step:
%zero.base = mul i32 %sum.slot, RECIPE_REGISTER_COUNT
%zero.index = add i32 %zero.base, %zero.r
%zero.ptr = getelementptr float, ptr %chunk.sums, i32 %zero.index
store float %state.zero, ptr %zero.ptr, align 4
%zero.next = add i32 %zero.r, 1
br label %chunk.zero.loop
k.begin:
%k.first = mul i32 %chunk, RECIPE_CHUNK_K
%k.limit.raw = add i32 %k.first, RECIPE_CHUNK_K
%k.over = icmp ugt i32 %k.limit.raw, %k.count
%k.limit = select i1 %k.over, i32 %k.count, i32 %k.limit.raw
%slot.sum.base = mul i32 %sum.slot, RECIPE_REGISTER_COUNT
%a.initial = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment(i32 %k.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.initial = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment(i32 %k.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %k.loop
k.loop:
%k = phi i32 [ %k.first, %k.begin ], [ %k.next, %register.done ]
%a.fragment = phi <RECIPE_REGISTER_M x i8> [ %a.initial, %k.begin ], [ %a.next, %register.done ]
%b.fragment = phi <RECIPE_REGISTER_N x i8> [ %b.initial, %k.begin ], [ %b.next, %register.done ]
%k.next = add i32 %k, 1
%k.more = icmp ult i32 %k.next, %k.limit
%k.prefetch = select i1 %k.more, i32 %k.next, i32 %k
%a.next = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment(i32 %k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.next = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment(i32 %k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%a.wide = call <RECIPE_REGISTER_M x float> @contraction_widen_m(<RECIPE_REGISTER_M x i8> %a.fragment)
br label %register.loop
register.loop:
%register.n = phi i32 [ 0, %k.loop ], [ %register.n.next, %register.next ]
%register.more = icmp ult i32 %register.n, RECIPE_REGISTER_N
br i1 %register.more, label %register.step, label %register.done
register.step:
%output.n.raw = add i32 %output.n.base, %register.n
%output.n.valid = icmp ult i32 %output.n.raw, %n.count
%b = extractelement <RECIPE_REGISTER_N x i8> %b.fragment, i32 %register.n
%b.wide = call float @recipe.decode(i8 %b)
%b.seed = insertelement <RECIPE_REGISTER_M x float> poison, float %b.wide, i32 0
%b.vector = shufflevector <RECIPE_REGISTER_M x float> %b.seed, <RECIPE_REGISTER_M x float> poison, <RECIPE_REGISTER_M x i32> zeroinitializer
%register.base = mul i32 %register.n, RECIPE_REGISTER_M
%sum.index = add i32 %slot.sum.base, %register.base
%sum.ptr = getelementptr float, ptr %chunk.sums, i32 %sum.index
%sum = load <RECIPE_REGISTER_M x float>, ptr %sum.ptr, align 4
%candidate = call <RECIPE_REGISTER_M x float> @recipe.state.madd.vector(<RECIPE_REGISTER_M x float> %sum, <RECIPE_REGISTER_M x float> %a.wide, <RECIPE_REGISTER_M x float> %b.vector)
store <RECIPE_REGISTER_M x float> %candidate, ptr %sum.ptr, align 4
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
%publish.r.more = icmp ult i32 %publish.r, RECIPE_REGISTER_COUNT
br i1 %publish.r.more, label %publish.sum.step, label %publish.finish
publish.sum.step:
%publish.source.base = mul i32 %publish.sum.slot, RECIPE_REGISTER_COUNT
%publish.source.index = add i32 %publish.source.base, %publish.r
%publish.source = getelementptr float, ptr %chunk.sums, i32 %publish.source.index
%publish.value = load float, ptr %publish.source, align 4
%publish.row = mul i32 %publish.chunk, %output.lanes
%publish.column = add i32 %publish.row, %output.lane
%publish.target.base = mul i32 %publish.column, RECIPE_REGISTER_COUNT
%publish.target.index = add i32 %publish.target.base, %publish.r
%publish.target = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %publish.target.index
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
%fold.r.more = icmp ult i32 %fold.r, RECIPE_REGISTER_COUNT
br i1 %fold.r.more, label %fold.sum.step, label %fold.finish
fold.sum.step:
%fold.row = mul i32 %fold.chunk, %output.lanes
%fold.column = add i32 %fold.row, %output.lane
%fold.source.base = mul i32 %fold.column, RECIPE_REGISTER_COUNT
%fold.source.index = add i32 %fold.source.base, %fold.r
%fold.source = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %fold.source.index
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
define internal void @contraction_vector_accumulate_bs(
ptr %sums, i1 %lane.active, i1 %lane.store, i32 %lid,
i32 %lane.k, i32 %k.lanes, i32 %output.lane, i32 %output.lanes,
i32 %output.m.base, i32 %output.n.base, i32 %m.count, i32 %n.count,
i32 %k.count, i32 %tile.m, i32 %tile.n, i32 %tile.k ) #1 {
entry:
%chunk.sums = alloca [RECIPE_CHUNK_VALUES x float], align 4
%state.zero = call float @recipe.state.from.u1(i1 false)
%chunks.adjusted = add i32 %k.count, RECIPE_CHUNK_K
%chunks.numerator = sub i32 %chunks.adjusted, 1
%chunks = udiv i32 %chunks.numerator, RECIPE_CHUNK_K
%single.sum.slot = icmp eq i32 RECIPE_CHUNK_VALUES, RECIPE_REGISTER_COUNT
%publish.lane.width = mul i32 %output.lanes, RECIPE_REGISTER_COUNT
%chunk.first = select i1 %lane.active, i32 %lane.k, i32 %chunks
%local.lane = icmp eq i32 %k.lanes, 1
%few.chunks = icmp ule i32 %chunks, 3
%local = or i1 %local.lane, %few.chunks
%local.owner = icmp eq i32 %lane.k, 0
%local.owner.active = and i1 %lane.active, %local.owner
br i1 %local, label %local.owner.check, label %shared.chunk.loop
local.owner.check:
br i1 %local.owner.active, label %local.k.begin, label %exit
; Use the same chunk boundaries and fold as the shared path, even when
; one lane computes every chunk. The schedule must not change rounding.
local.k.begin:
br label %local.chunk.loop
local.chunk.loop:
%local.chunk.first = phi i32 [ 0, %local.k.begin ], [ %local.chunk.limit, %local.store ]
%local.chunk.end = add i32 %local.chunk.first, RECIPE_CHUNK_K
%local.chunk.over = icmp ugt i32 %local.chunk.end, %k.count
%local.chunk.limit = select i1 %local.chunk.over, i32 %k.count, i32 %local.chunk.end
%local.sums.prior = load <RECIPE_REGISTER_COUNT x float>, ptr %sums, align 4
%local.a.initial = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment_vector(i32 %local.chunk.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.initial = call <RECIPE_REGISTER_N x float> @contraction_b_fragment_state(i32 %local.chunk.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %local.k.loop
local.k.loop:
%local.k = phi i32 [ %local.chunk.first, %local.chunk.loop ], [ %local.k.next, %local.product.done ]
%local.sums = phi <RECIPE_REGISTER_COUNT x float> [ zeroinitializer, %local.chunk.loop ], [ %local.sums.current, %local.product.done ]
%local.a.fragment = phi <RECIPE_REGISTER_M x i8> [ %local.a.initial, %local.chunk.loop ], [ %local.a.next, %local.product.done ]
%local.b.fragment = phi <RECIPE_REGISTER_N x float> [ %local.b.initial, %local.chunk.loop ], [ %local.b.next, %local.product.done ]
%local.k.next = add i32 %local.k, 1
%local.k.more = icmp ult i32 %local.k.next, %local.chunk.limit
%local.k.prefetch = select i1 %local.k.more, i32 %local.k.next, i32 %local.k
%local.a.next = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment_vector(i32 %local.k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.next = call <RECIPE_REGISTER_N x float> @contraction_b_fragment_state(i32 %local.k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%local.a.wide = call <RECIPE_REGISTER_M x float> @contraction_widen_m(<RECIPE_REGISTER_M x i8> %local.a.fragment)
br label %local.product.loop
local.product.loop:
%local.product = phi i32 [ 0, %local.k.loop ], [ %local.product.next, %local.product.step ]
%local.sums.current = phi <RECIPE_REGISTER_COUNT x float> [ %local.sums, %local.k.loop ], [ %local.candidate, %local.product.step ]
%local.product.more = icmp ult i32 %local.product, RECIPE_REGISTER_COUNT
br i1 %local.product.more, label %local.product.step, label %local.product.done
local.product.step:
%local.a.index = urem i32 %local.product, RECIPE_REGISTER_M
%local.b.index = udiv i32 %local.product, RECIPE_REGISTER_M
%local.a = extractelement <RECIPE_REGISTER_M x float> %local.a.wide, i32 %local.a.index
%local.b.wide = extractelement <RECIPE_REGISTER_N x float> %local.b.fragment, i32 %local.b.index
%local.sum = extractelement <RECIPE_REGISTER_COUNT x float> %local.sums.current, i32 %local.product
%local.value = call float @recipe.state.madd(float %local.sum, float %local.a, float %local.b.wide)
%local.candidate = insertelement <RECIPE_REGISTER_COUNT x float> %local.sums.current, float %local.value, i32 %local.product
%local.product.next = add i32 %local.product, 1
br label %local.product.loop, !llvm.loop !0
local.product.done:
br i1 %local.k.more, label %local.k.loop, label %local.store
local.store:
%local.folded = fadd <RECIPE_REGISTER_COUNT x float> %local.sums.prior, %local.sums.current
store <RECIPE_REGISTER_COUNT x float> %local.folded, ptr %sums, align 4
%local.chunk.more = icmp ult i32 %local.chunk.limit, %k.count
br i1 %local.chunk.more, label %local.chunk.loop, label %exit
shared.chunk.loop:
%chunk = phi i32 [ %chunk.first, %entry ], [ %chunk.next, %chunk.finish ]
%slot = phi i32 [ 0, %entry ], [ %slot.next, %chunk.finish ]
%sum.slot = select i1 %single.sum.slot, i32 0, i32 %slot
%chunk.more = icmp ult i32 %chunk, %chunks
br i1 %chunk.more, label %chunk.zero.loop, label %chunk.done
chunk.zero.loop:
%zero.r = phi i32 [ 0, %shared.chunk.loop ], [ %zero.next, %chunk.zero.step ]
%zero.more = icmp ult i32 %zero.r, RECIPE_REGISTER_COUNT
br i1 %zero.more, label %chunk.zero.step, label %k.begin
chunk.zero.step:
%zero.base = mul i32 %sum.slot, RECIPE_REGISTER_COUNT
%zero.index = add i32 %zero.base, %zero.r
%zero.ptr = getelementptr float, ptr %chunk.sums, i32 %zero.index
store float %state.zero, ptr %zero.ptr, align 4
%zero.next = add i32 %zero.r, 1
br label %chunk.zero.loop
k.begin:
%k.first = mul i32 %chunk, RECIPE_CHUNK_K
%k.limit.raw = add i32 %k.first, RECIPE_CHUNK_K
%k.over = icmp ugt i32 %k.limit.raw, %k.count
%k.limit = select i1 %k.over, i32 %k.count, i32 %k.limit.raw
%slot.sum.base = mul i32 %sum.slot, RECIPE_REGISTER_COUNT
%a.initial = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment_vector(i32 %k.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.initial = call <RECIPE_REGISTER_N x float> @contraction_b_fragment_state(i32 %k.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %k.loop
k.loop:
%k = phi i32 [ %k.first, %k.begin ], [ %k.next, %register.done ]
%a.fragment = phi <RECIPE_REGISTER_M x i8> [ %a.initial, %k.begin ], [ %a.next, %register.done ]
%b.fragment = phi <RECIPE_REGISTER_N x float> [ %b.initial, %k.begin ], [ %b.next, %register.done ]
%k.next = add i32 %k, 1
%k.more = icmp ult i32 %k.next, %k.limit
%k.prefetch = select i1 %k.more, i32 %k.next, i32 %k
%a.next = call <RECIPE_REGISTER_M x i8> @contraction_a_fragment_vector(i32 %k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.next = call <RECIPE_REGISTER_N x float> @contraction_b_fragment_state(i32 %k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%a.wide = call <RECIPE_REGISTER_M x float> @contraction_widen_m(<RECIPE_REGISTER_M x i8> %a.fragment)
br label %register.loop
register.loop:
%register.n = phi i32 [ 0, %k.loop ], [ %register.n.next, %register.next ]
%register.more = icmp ult i32 %register.n, RECIPE_REGISTER_N
br i1 %register.more, label %register.step, label %register.done
register.step:
%output.n.raw = add i32 %output.n.base, %register.n
%output.n.valid = icmp ult i32 %output.n.raw, %n.count
%b.wide = extractelement <RECIPE_REGISTER_N x float> %b.fragment, i32 %register.n
%b.seed = insertelement <RECIPE_REGISTER_M x float> poison, float %b.wide, i32 0
%b.vector = shufflevector <RECIPE_REGISTER_M x float> %b.seed, <RECIPE_REGISTER_M x float> poison, <RECIPE_REGISTER_M x i32> zeroinitializer
%register.base = mul i32 %register.n, RECIPE_REGISTER_M
%sum.index = add i32 %slot.sum.base, %register.base
%sum.ptr = getelementptr float, ptr %chunk.sums, i32 %sum.index
%sum = load <RECIPE_REGISTER_M x float>, ptr %sum.ptr, align 4
%candidate = call <RECIPE_REGISTER_M x float> @recipe.state.madd.vector(<RECIPE_REGISTER_M x float> %sum, <RECIPE_REGISTER_M x float> %a.wide, <RECIPE_REGISTER_M x float> %b.vector)
store <RECIPE_REGISTER_M x float> %candidate, ptr %sum.ptr, align 4
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
%publish.r.more = icmp ult i32 %publish.r, RECIPE_REGISTER_COUNT
br i1 %publish.r.more, label %publish.sum.step, label %publish.finish
publish.sum.step:
%publish.source.base = mul i32 %publish.sum.slot, RECIPE_REGISTER_COUNT
%publish.source.index = add i32 %publish.source.base, %publish.r
%publish.source = getelementptr float, ptr %chunk.sums, i32 %publish.source.index
%publish.value = load float, ptr %publish.source, align 4
%publish.row = mul i32 %publish.chunk, %output.lanes
%publish.column = add i32 %publish.row, %output.lane
%publish.target.base = mul i32 %publish.column, RECIPE_REGISTER_COUNT
%publish.target.index = add i32 %publish.target.base, %publish.r
%publish.target = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %publish.target.index
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
%fold.r.more = icmp ult i32 %fold.r, RECIPE_REGISTER_COUNT
br i1 %fold.r.more, label %fold.sum.step, label %fold.finish
fold.sum.step:
%fold.row = mul i32 %fold.chunk, %output.lanes
%fold.column = add i32 %fold.row, %output.lane
%fold.source.base = mul i32 %fold.column, RECIPE_REGISTER_COUNT
%fold.source.index = add i32 %fold.source.base, %fold.r
%fold.source = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %fold.source.index
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
define internal void @contraction_vector_accumulate_as(
ptr %sums, i1 %lane.active, i1 %lane.store, i32 %lid,
i32 %lane.k, i32 %k.lanes, i32 %output.lane, i32 %output.lanes,
i32 %output.m.base, i32 %output.n.base, i32 %m.count, i32 %n.count,
i32 %k.count, i32 %tile.m, i32 %tile.n, i32 %tile.k ) #1 {
entry:
%chunk.sums = alloca [RECIPE_CHUNK_VALUES x float], align 4
%state.zero = call float @recipe.state.from.u1(i1 false)
%chunks.adjusted = add i32 %k.count, RECIPE_CHUNK_K
%chunks.numerator = sub i32 %chunks.adjusted, 1
%chunks = udiv i32 %chunks.numerator, RECIPE_CHUNK_K
%single.sum.slot = icmp eq i32 RECIPE_CHUNK_VALUES, RECIPE_REGISTER_COUNT
%publish.lane.width = mul i32 %output.lanes, RECIPE_REGISTER_COUNT
%chunk.first = select i1 %lane.active, i32 %lane.k, i32 %chunks
%local.lane = icmp eq i32 %k.lanes, 1
%few.chunks = icmp ule i32 %chunks, 3
%local = or i1 %local.lane, %few.chunks
%local.owner = icmp eq i32 %lane.k, 0
%local.owner.active = and i1 %lane.active, %local.owner
br i1 %local, label %local.owner.check, label %shared.chunk.loop
local.owner.check:
br i1 %local.owner.active, label %local.k.begin, label %exit
; Use the same chunk boundaries and fold as the shared path, even when
; one lane computes every chunk. The schedule must not change rounding.
local.k.begin:
br label %local.chunk.loop
local.chunk.loop:
%local.chunk.first = phi i32 [ 0, %local.k.begin ], [ %local.chunk.limit, %local.store ]
%local.chunk.end = add i32 %local.chunk.first, RECIPE_CHUNK_K
%local.chunk.over = icmp ugt i32 %local.chunk.end, %k.count
%local.chunk.limit = select i1 %local.chunk.over, i32 %k.count, i32 %local.chunk.end
%local.sums.prior = load <RECIPE_REGISTER_COUNT x float>, ptr %sums, align 4
%local.a.initial = call <RECIPE_REGISTER_M x float> @contraction_a_fragment_state(i32 %local.chunk.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.initial = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment_after_state(i32 %local.chunk.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %local.k.loop
local.k.loop:
%local.k = phi i32 [ %local.chunk.first, %local.chunk.loop ], [ %local.k.next, %local.product.done ]
%local.sums = phi <RECIPE_REGISTER_COUNT x float> [ zeroinitializer, %local.chunk.loop ], [ %local.sums.current, %local.product.done ]
%local.a.fragment = phi <RECIPE_REGISTER_M x float> [ %local.a.initial, %local.chunk.loop ], [ %local.a.next, %local.product.done ]
%local.b.fragment = phi <RECIPE_REGISTER_N x i8> [ %local.b.initial, %local.chunk.loop ], [ %local.b.next, %local.product.done ]
%local.k.next = add i32 %local.k, 1
%local.k.more = icmp ult i32 %local.k.next, %local.chunk.limit
%local.k.prefetch = select i1 %local.k.more, i32 %local.k.next, i32 %local.k
%local.a.next = call <RECIPE_REGISTER_M x float> @contraction_a_fragment_state(i32 %local.k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.next = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment_after_state(i32 %local.k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%local.a.wide = select i1 true, <RECIPE_REGISTER_M x float> %local.a.fragment, <RECIPE_REGISTER_M x float> %local.a.fragment
br label %local.product.loop
local.product.loop:
%local.product = phi i32 [ 0, %local.k.loop ], [ %local.product.next, %local.product.step ]
%local.sums.current = phi <RECIPE_REGISTER_COUNT x float> [ %local.sums, %local.k.loop ], [ %local.candidate, %local.product.step ]
%local.product.more = icmp ult i32 %local.product, RECIPE_REGISTER_COUNT
br i1 %local.product.more, label %local.product.step, label %local.product.done
local.product.step:
%local.a.index = urem i32 %local.product, RECIPE_REGISTER_M
%local.b.index = udiv i32 %local.product, RECIPE_REGISTER_M
%local.a = extractelement <RECIPE_REGISTER_M x float> %local.a.wide, i32 %local.a.index
%local.b = extractelement <RECIPE_REGISTER_N x i8> %local.b.fragment, i32 %local.b.index
%local.b.wide = call float @recipe.decode(i8 %local.b)
%local.sum = extractelement <RECIPE_REGISTER_COUNT x float> %local.sums.current, i32 %local.product
%local.value = call float @recipe.state.madd(float %local.sum, float %local.a, float %local.b.wide)
%local.candidate = insertelement <RECIPE_REGISTER_COUNT x float> %local.sums.current, float %local.value, i32 %local.product
%local.product.next = add i32 %local.product, 1
br label %local.product.loop, !llvm.loop !0
local.product.done:
br i1 %local.k.more, label %local.k.loop, label %local.store
local.store:
%local.folded = fadd <RECIPE_REGISTER_COUNT x float> %local.sums.prior, %local.sums.current
store <RECIPE_REGISTER_COUNT x float> %local.folded, ptr %sums, align 4
%local.chunk.more = icmp ult i32 %local.chunk.limit, %k.count
br i1 %local.chunk.more, label %local.chunk.loop, label %exit
shared.chunk.loop:
%chunk = phi i32 [ %chunk.first, %entry ], [ %chunk.next, %chunk.finish ]
%slot = phi i32 [ 0, %entry ], [ %slot.next, %chunk.finish ]
%sum.slot = select i1 %single.sum.slot, i32 0, i32 %slot
%chunk.more = icmp ult i32 %chunk, %chunks
br i1 %chunk.more, label %chunk.zero.loop, label %chunk.done
chunk.zero.loop:
%zero.r = phi i32 [ 0, %shared.chunk.loop ], [ %zero.next, %chunk.zero.step ]
%zero.more = icmp ult i32 %zero.r, RECIPE_REGISTER_COUNT
br i1 %zero.more, label %chunk.zero.step, label %k.begin
chunk.zero.step:
%zero.base = mul i32 %sum.slot, RECIPE_REGISTER_COUNT
%zero.index = add i32 %zero.base, %zero.r
%zero.ptr = getelementptr float, ptr %chunk.sums, i32 %zero.index
store float %state.zero, ptr %zero.ptr, align 4
%zero.next = add i32 %zero.r, 1
br label %chunk.zero.loop
k.begin:
%k.first = mul i32 %chunk, RECIPE_CHUNK_K
%k.limit.raw = add i32 %k.first, RECIPE_CHUNK_K
%k.over = icmp ugt i32 %k.limit.raw, %k.count
%k.limit = select i1 %k.over, i32 %k.count, i32 %k.limit.raw
%slot.sum.base = mul i32 %sum.slot, RECIPE_REGISTER_COUNT
%a.initial = call <RECIPE_REGISTER_M x float> @contraction_a_fragment_state(i32 %k.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.initial = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment_after_state(i32 %k.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %k.loop
k.loop:
%k = phi i32 [ %k.first, %k.begin ], [ %k.next, %register.done ]
%a.fragment = phi <RECIPE_REGISTER_M x float> [ %a.initial, %k.begin ], [ %a.next, %register.done ]
%b.fragment = phi <RECIPE_REGISTER_N x i8> [ %b.initial, %k.begin ], [ %b.next, %register.done ]
%k.next = add i32 %k, 1
%k.more = icmp ult i32 %k.next, %k.limit
%k.prefetch = select i1 %k.more, i32 %k.next, i32 %k
%a.next = call <RECIPE_REGISTER_M x float> @contraction_a_fragment_state(i32 %k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.next = call <RECIPE_REGISTER_N x i8> @contraction_b_fragment_after_state(i32 %k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%a.wide = select i1 true, <RECIPE_REGISTER_M x float> %a.fragment, <RECIPE_REGISTER_M x float> %a.fragment
br label %register.loop
register.loop:
%register.n = phi i32 [ 0, %k.loop ], [ %register.n.next, %register.next ]
%register.more = icmp ult i32 %register.n, RECIPE_REGISTER_N
br i1 %register.more, label %register.step, label %register.done
register.step:
%output.n.raw = add i32 %output.n.base, %register.n
%output.n.valid = icmp ult i32 %output.n.raw, %n.count
%b = extractelement <RECIPE_REGISTER_N x i8> %b.fragment, i32 %register.n
%b.wide = call float @recipe.decode(i8 %b)
%b.seed = insertelement <RECIPE_REGISTER_M x float> poison, float %b.wide, i32 0
%b.vector = shufflevector <RECIPE_REGISTER_M x float> %b.seed, <RECIPE_REGISTER_M x float> poison, <RECIPE_REGISTER_M x i32> zeroinitializer
%register.base = mul i32 %register.n, RECIPE_REGISTER_M
%sum.index = add i32 %slot.sum.base, %register.base
%sum.ptr = getelementptr float, ptr %chunk.sums, i32 %sum.index
%sum = load <RECIPE_REGISTER_M x float>, ptr %sum.ptr, align 4
%candidate = call <RECIPE_REGISTER_M x float> @recipe.state.madd.vector(<RECIPE_REGISTER_M x float> %sum, <RECIPE_REGISTER_M x float> %a.wide, <RECIPE_REGISTER_M x float> %b.vector)
store <RECIPE_REGISTER_M x float> %candidate, ptr %sum.ptr, align 4
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
%publish.r.more = icmp ult i32 %publish.r, RECIPE_REGISTER_COUNT
br i1 %publish.r.more, label %publish.sum.step, label %publish.finish
publish.sum.step:
%publish.source.base = mul i32 %publish.sum.slot, RECIPE_REGISTER_COUNT
%publish.source.index = add i32 %publish.source.base, %publish.r
%publish.source = getelementptr float, ptr %chunk.sums, i32 %publish.source.index
%publish.value = load float, ptr %publish.source, align 4
%publish.row = mul i32 %publish.chunk, %output.lanes
%publish.column = add i32 %publish.row, %output.lane
%publish.target.base = mul i32 %publish.column, RECIPE_REGISTER_COUNT
%publish.target.index = add i32 %publish.target.base, %publish.r
%publish.target = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %publish.target.index
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
%fold.r.more = icmp ult i32 %fold.r, RECIPE_REGISTER_COUNT
br i1 %fold.r.more, label %fold.sum.step, label %fold.finish
fold.sum.step:
%fold.row = mul i32 %fold.chunk, %output.lanes
%fold.column = add i32 %fold.row, %output.lane
%fold.source.base = mul i32 %fold.column, RECIPE_REGISTER_COUNT
%fold.source.index = add i32 %fold.source.base, %fold.r
%fold.source = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %fold.source.index
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
%sum.ptr = getelementptr [RECIPE_REGISTER_N x float], ptr %sums, i32 0, i32 %slot
%previous = load float, ptr %sum.ptr, align 4 %initial = select i1 %first, float %zero, float %previous br label %r.loop
r.loop:
%r = phi i32 [ 0, %channel.begin ], [ %r.next, %r.step ] %sum = phi float [ %initial, %channel.begin ], [ %sum.next, %r.step ]
%r.more = icmp ult i32 %r, %r.count br i1 %r.more, label %r.step, label %sum.store
r.step:
%base = mul i32 %tile.m, %tile.k %local = call i32 @contraction_vector_b_index(i32 %r, i32 %channel, i32 %tile.n, i32 %tile.k) %index = add i32 %base, %local
%ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %index
%raw = load i8, ptr addrspace(3) %ptr, align 1 %value = call float @recipe.decode(i8 %raw) %sum.next = call float @recipe.state.add(float %sum, float %value)
%r.next = add i32 %r, 1 br label %r.loop
sum.store:
store float %sum, ptr %sum.ptr, align 4 br i1 %last, label %destination.store, label %channel.done
destination.store:
%filter = add i32 %n.base, %channel %bias.base = mul i32 %out.channels, %window %bias.local = add i32 %bias.base, %filter %bias.index = add i32 %store.offset, %bias.local
%bias.ptr = getelementptr inbounds i8, ptr addrspace(1) %destination, i32 %bias.index
%bias = call i8 @recipe.encode(float %sum) store i8 %bias, ptr addrspace(1) %bias.ptr, align 1 br label %channel.done
channel.done:
%channel.next = add i32 %channel, %block %slot.next = add i32 %slot, 1 br label %channel.loop
exit: ret void
}
define internal void @contraction_bias_accumulate_state(
ptr %sums, ptr addrspace(1) %destination,
i1 %enable, i1 %first, i1 %last, i32 %lid, i32 %block,
i32 %n.base, i32 %n.count, i32 %r.count, i32 %out.channels, i32 %window,
i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %store.offset ) #1 { entry:
%zero = call float @recipe.state.from.u1(i1 false) br i1 %enable, label %channel.loop, label %exit
channel.loop:
%channel = phi i32 [ %lid, %entry ], [ %channel.next, %channel.done ] %slot = phi i32 [ 0, %entry ], [ %slot.next, %channel.done ]
%channel.more = icmp ult i32 %channel, %n.count br i1 %channel.more, label %channel.begin, label %exit
channel.begin:
%sum.ptr = getelementptr [RECIPE_REGISTER_N x float], ptr %sums, i32 0, i32 %slot
%previous = load float, ptr %sum.ptr, align 4 %initial = select i1 %first, float %zero, float %previous br label %r.loop
r.loop:
%r = phi i32 [ 0, %channel.begin ], [ %r.next, %r.step ] %sum = phi float [ %initial, %channel.begin ], [ %sum.next, %r.step ]
%r.more = icmp ult i32 %r, %r.count br i1 %r.more, label %r.step, label %sum.store
r.step:
%a.elements = mul i32 %tile.m, %tile.k %base = call i32 @contraction_state_after_model(i32 %a.elements) %local = call i32 @contraction_vector_b_index(i32 %r, i32 %channel, i32 %tile.n, i32 %tile.k) %index = add i32 %base, %local
%ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %index
%value = load float, ptr addrspace(3) %ptr, align 4 %sum.next = call float @recipe.state.add(float %sum, float %value)
%r.next = add i32 %r, 1 br label %r.loop
sum.store:
store float %sum, ptr %sum.ptr, align 4 br i1 %last, label %destination.store, label %channel.done
destination.store:
%filter = add i32 %n.base, %channel %bias.base = mul i32 %out.channels, %window %bias.local = add i32 %bias.base, %filter %bias.index = add i32 %store.offset, %bias.local
%bias.ptr = getelementptr inbounds float, ptr addrspace(1) %destination, i32 %bias.index
store float %sum, ptr addrspace(1) %bias.ptr, align 4 br label %channel.done
channel.done:
%channel.next = add i32 %channel, %block %slot.next = add i32 %slot, 1 br label %channel.loop
exit: ret void
}
; Widen a staged A fragment to the arithmetic type once per K step, so the inner
; register loop never converts.
define internal <RECIPE_REGISTER_M x float> @contraction_widen_m(<RECIPE_REGISTER_M x i8> %source) #1 {
entry:
br label %loop
loop:
%p = phi i32 [ 0, %entry ], [ %p.next, %step ]
%result = phi <RECIPE_REGISTER_M x float> [ poison, %entry ], [ %next, %step ]
%more = icmp ult i32 %p, RECIPE_REGISTER_M
br i1 %more, label %step, label %done
step:
%value = extractelement <RECIPE_REGISTER_M x i8> %source, i32 %p
%wide = call float @recipe.decode(i8 %value)
%next = insertelement <RECIPE_REGISTER_M x float> %result, float %wide, i32 %p
%p.next = add i32 %p, 1
br label %loop
done:
ret <RECIPE_REGISTER_M x float> %result
}


declare <8 x float> @recipe.wmma(<16 x i8>, <16 x i8>, <8 x float>)
; Matrix arithmetic consumes the operands staged by the common contraction
; composer, writes one state-width partial tile beside them, then maps that tile
; back into the composer's register ownership. Scheduling, tails, K tiling, and
; the epilogue therefore remain identical to the vector method.
define internal void @contraction_matrix_accumulate(
ptr %sums, i1 %lane.active, i1 %lane.store, i32 %lid,
i32 %lane.k, i32 %k.lanes, i32 %output.lane, i32 %output.lanes,
i32 %output.m.base, i32 %output.n.base, i32 %m.count, i32 %n.count,
i32 %k.count, i32 %tile.m, i32 %tile.n, i32 %tile.k ) #1 { entry:
%wave = udiv i32 %lid, 32
%lane = urem i32 %lid, 32
%lane.local = urem i32 %lane, 16
%m.wave = mul i32 %wave, 16
%m = add i32 %m.wave, %lane.local
%n.first = add i32 %lane.local, 0
%n.second = add i32 %lane.local, 16
%n.third = add i32 %lane.local, 32
%n.fourth = add i32 %lane.local, 48
%m.valid = icmp ult i32 %m, %m.count
%n.first.valid = icmp ult i32 %n.first, %n.count
%n.second.valid = icmp ult i32 %n.second, %n.count
%n.third.valid = icmp ult i32 %n.third, %n.count
%n.fourth.valid = icmp ult i32 %n.fourth, %n.count
%m.safe = select i1 %m.valid, i32 %m, i32 0
%n.first.safe = select i1 %n.first.valid, i32 %n.first, i32 0
%n.second.safe = select i1 %n.second.valid, i32 %n.second, i32 0
%n.third.safe = select i1 %n.third.valid, i32 %n.third, i32 0
%n.fourth.safe = select i1 %n.fourth.valid, i32 %n.fourth, i32 0
%matrix.k.adjusted = add i32 %k.count, 15
%matrix.k.rounded = and i32 %matrix.k.adjusted, -16
br label %matrix.k.loop
matrix.k.loop:
%matrix.k = phi i32 [ 0, %entry ], [ %matrix.k.next, %matrix.k.done ]
%matrix.first = phi <8 x float> [ zeroinitializer, %entry ], [ %matrix.first.next, %matrix.k.done ]
%matrix.second = phi <8 x float> [ zeroinitializer, %entry ], [ %matrix.second.next, %matrix.k.done ]
%matrix.third = phi <8 x float> [ zeroinitializer, %entry ], [ %matrix.third.next, %matrix.k.done ]
%matrix.fourth = phi <8 x float> [ zeroinitializer, %entry ], [ %matrix.fourth.next, %matrix.k.done ]
%matrix.k.more = icmp ult i32 %matrix.k, %matrix.k.rounded
br i1 %matrix.k.more, label %matrix.full, label %matrix.store.loop
matrix.full:
%matrix.a.index = call i32 @contraction_vector_a_index(i32 %matrix.k, i32 %m.safe, i32 %tile.m, i32 %tile.k)
%matrix.a.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.a.index
%matrix.a.loaded = load <16 x i8>, ptr addrspace(3) %matrix.a.ptr, align 16
%matrix.a.full = select i1 %m.valid, <16 x i8> %matrix.a.loaded, <16 x i8> zeroinitializer
%matrix.b.base = mul i32 %tile.m, %tile.k
%matrix.b.first.local = call i32 @contraction_vector_b_index(i32 %matrix.k, i32 %n.first.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.first.index = add i32 %matrix.b.base, %matrix.b.first.local
%matrix.b.first.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.b.first.index
%matrix.b.first.loaded = load <16 x i8>, ptr addrspace(3) %matrix.b.first.ptr, align 16
%matrix.b.first.full = select i1 %n.first.valid, <16 x i8> %matrix.b.first.loaded, <16 x i8> zeroinitializer
%matrix.b.second.local = call i32 @contraction_vector_b_index(i32 %matrix.k, i32 %n.second.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.second.index = add i32 %matrix.b.base, %matrix.b.second.local
%matrix.b.second.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.b.second.index
%matrix.b.second.loaded = load <16 x i8>, ptr addrspace(3) %matrix.b.second.ptr, align 16
%matrix.b.second.full = select i1 %n.second.valid, <16 x i8> %matrix.b.second.loaded, <16 x i8> zeroinitializer
%matrix.b.third.local = call i32 @contraction_vector_b_index(i32 %matrix.k, i32 %n.third.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.third.index = add i32 %matrix.b.base, %matrix.b.third.local
%matrix.b.third.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.b.third.index
%matrix.b.third.loaded = load <16 x i8>, ptr addrspace(3) %matrix.b.third.ptr, align 16
%matrix.b.third.full = select i1 %n.third.valid, <16 x i8> %matrix.b.third.loaded, <16 x i8> zeroinitializer
%matrix.b.fourth.local = call i32 @contraction_vector_b_index(i32 %matrix.k, i32 %n.fourth.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.fourth.index = add i32 %matrix.b.base, %matrix.b.fourth.local
%matrix.b.fourth.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.b.fourth.index
%matrix.b.fourth.loaded = load <16 x i8>, ptr addrspace(3) %matrix.b.fourth.ptr, align 16
%matrix.b.fourth.full = select i1 %n.fourth.valid, <16 x i8> %matrix.b.fourth.loaded, <16 x i8> zeroinitializer
%matrix.first.next = call <8 x float> @recipe.wmma(<16 x i8> %matrix.a.full, <16 x i8> %matrix.b.first.full, <8 x float> %matrix.first)
%matrix.second.next = call <8 x float> @recipe.wmma(<16 x i8> %matrix.a.full, <16 x i8> %matrix.b.second.full, <8 x float> %matrix.second)
%matrix.third.next = call <8 x float> @recipe.wmma(<16 x i8> %matrix.a.full, <16 x i8> %matrix.b.third.full, <8 x float> %matrix.third)
%matrix.fourth.next = call <8 x float> @recipe.wmma(<16 x i8> %matrix.a.full, <16 x i8> %matrix.b.fourth.full, <8 x float> %matrix.fourth)
br label %matrix.k.done
matrix.k.done:
%matrix.k.next = add i32 %matrix.k, 16
br label %matrix.k.loop
matrix.store.loop:
%matrix.register = phi i32 [ 0, %matrix.k.loop ], [ %matrix.register.next, %matrix.store.step ]
%matrix.register.more = icmp ult i32 %matrix.register, 8
br i1 %matrix.register.more, label %matrix.store.step, label %matrix.exit
matrix.store.step:
%matrix.first.value = extractelement <8 x float> %matrix.first, i32 %matrix.register
%matrix.first.target = getelementptr float, ptr %sums, i32 %matrix.register
%matrix.first.current = load float, ptr %matrix.first.target, align 4
%matrix.first.sum = call float @recipe.state.add(float %matrix.first.current, float %matrix.first.value)
store float %matrix.first.sum, ptr %matrix.first.target, align 4
%matrix.second.register = add i32 %matrix.register, 8
%matrix.second.value = extractelement <8 x float> %matrix.second, i32 %matrix.register
%matrix.second.target = getelementptr float, ptr %sums, i32 %matrix.second.register
%matrix.second.current = load float, ptr %matrix.second.target, align 4
%matrix.second.sum = call float @recipe.state.add(float %matrix.second.current, float %matrix.second.value)
store float %matrix.second.sum, ptr %matrix.second.target, align 4
%matrix.third.register = add i32 %matrix.register, 16
%matrix.third.value = extractelement <8 x float> %matrix.third, i32 %matrix.register
%matrix.third.target = getelementptr float, ptr %sums, i32 %matrix.third.register
%matrix.third.current = load float, ptr %matrix.third.target, align 4
%matrix.third.sum = call float @recipe.state.add(float %matrix.third.current, float %matrix.third.value)
store float %matrix.third.sum, ptr %matrix.third.target, align 4
%matrix.fourth.register = add i32 %matrix.register, 24
%matrix.fourth.value = extractelement <8 x float> %matrix.fourth, i32 %matrix.register
%matrix.fourth.target = getelementptr float, ptr %sums, i32 %matrix.fourth.register
%matrix.fourth.current = load float, ptr %matrix.fourth.target, align 4
%matrix.fourth.sum = call float @recipe.state.add(float %matrix.fourth.current, float %matrix.fourth.value)
store float %matrix.fourth.sum, ptr %matrix.fourth.target, align 4
%matrix.register.next = add i32 %matrix.register, 1
br label %matrix.store.loop
matrix.exit:
ret void
}
; The model compiler replaces this with one arm per packed node.
define internal i8 @recipe.model.decode(ptr addrspace(1) %matrix, i64 %index, i32 %node) #1 { entry: unreachable }
; One weight of a node-relative span: a dense node loads it, a packed node decodes it.
define internal i8 @recipe.model.weight(ptr addrspace(1) %weights, i64 %index, i32 %decode) #1 { entry:
%packed = icmp ne i32 %decode, 0 br i1 %packed, label %decoded, label %direct
direct: %ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %index %loaded = load i8, ptr addrspace(1) %ptr, align 1 ret i8 %loaded
decoded: %value = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %index, i32 %decode) ret i8 %value }
; Single-position GEMV. Decode appends one position at a time, so one output
; row can be assigned to each lane without staging a full input vector or
; synchronizing every K tile. The wrapper below selects this only for the
; exact one-position, non-activation forward case.
define internal void @contraction_forward_gemv_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel,
i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode ) #1 { entry:
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%groups = udiv i32 %threads, %block
%state.zero = call float @recipe.state.from.u1(i1 false)
%terms = add i32 %in.channels, 0
%terms.wide = zext i32 %terms to i64
%weight.base.wide = add i64 %weight.base, 0
%position = zext i32 %out.begin to i64
%tiles.adjusted = add i32 %out.channels, %block
%tiles.numerator = sub i32 %tiles.adjusted, 1
%tiles = udiv i32 %tiles.numerator, %block
br label %job.loop
job.loop:
%job = phi i32 [ %group, %entry ], [ %job.next, %job.done ]
%job.more = icmp ult i32 %job, %tiles
br i1 %job.more, label %job.step, label %exit
job.step:
%n.base = mul i32 %job, %block
%n.remaining = sub i32 %out.channels, %n.base
%n.partial = icmp ult i32 %n.remaining, %block
%n.count = select i1 %n.partial, i32 %n.remaining, i32 %block
%lane.active = icmp ult i32 %lid, %n.count
%channel = add i32 %n.base, %lid
%channel.wide = zext i32 %channel to i64
br i1 %lane.active, label %sum.loop, label %job.done
sum.loop:
%k = phi i32 [ 0, %job.step ], [ %k.next, %weight.ready ]
%partials = phi <32 x float> [ zeroinitializer, %job.step ], [ %partials.next, %weight.ready ]
%k.more = icmp ult i32 %k, %terms
br i1 %k.more, label %sum.step, label %sum.done
sum.step:
%k.wide = zext i32 %k to i64
%weight.local = mul i64 %channel.wide, %terms.wide
%weight.local.index = add i64 %weight.local, %k.wide
%weight.decode.index = add i64 %weight.base.wide, %weight.local.index
%weight.packed = icmp ne i32 %decode, 0
br i1 %weight.packed, label %weight.packed.load, label %weight.dense.load
weight.dense.load:
%weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %weight.local.index
%weight.dense.model = load i8, ptr addrspace(1) %weight.ptr, align 1
br label %weight.ready
weight.packed.load:
%weight.packed.model = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %weight.decode.index, i32 %decode)
br label %weight.ready
weight.ready:
%weight.model = phi i8 [ %weight.dense.model, %weight.dense.load ], [ %weight.packed.model, %weight.packed.load ]
%input.channel = zext i32 %k to i64
%input.length.wide = zext i32 %in.length to i64
%input.offset = mul i64 %input.channel, %input.length.wide
%input.index = add i64 %input.offset, %position
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %input.index
%input.model = load i8, ptr addrspace(1) %input.ptr, align 1
%weight.wide = call float @recipe.decode(i8 %weight.model)
%input.wide = call float @recipe.decode(i8 %input.model)
%product = call float @recipe.state.mul(float %weight.wide, float %input.wide)
%partial.lane = and i32 %k, 31
%partial = extractelement <32 x float> %partials, i32 %partial.lane
%partial.next = call float @recipe.state.add(float %partial, float %product)
%partials.next = insertelement <32 x float> %partials, float %partial.next, i32 %partial.lane
%k.next = add i32 %k, 1
br label %sum.loop
sum.done:
; Match the 32-lane dot reduction without depending on physical wave width.
%partial.partners.16 = shufflevector <32 x float> %partials, <32 x float> poison, <32 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
%partial.sums.16 = fadd <32 x float> %partials, %partial.partners.16
%partial.partners.8 = shufflevector <32 x float> %partial.sums.16, <32 x float> poison, <32 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
%partial.sums.8 = fadd <32 x float> %partial.sums.16, %partial.partners.8
%partial.partners.4 = shufflevector <32 x float> %partial.sums.8, <32 x float> poison, <32 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 20, i32 21, i32 22, i32 23, i32 16, i32 17, i32 18, i32 19, i32 28, i32 29, i32 30, i32 31, i32 24, i32 25, i32 26, i32 27>
%partial.sums.4 = fadd <32 x float> %partial.sums.8, %partial.partners.4
%partial.partners.2 = shufflevector <32 x float> %partial.sums.4, <32 x float> poison, <32 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9, i32 14, i32 15, i32 12, i32 13, i32 18, i32 19, i32 16, i32 17, i32 22, i32 23, i32 20, i32 21, i32 26, i32 27, i32 24, i32 25, i32 30, i32 31, i32 28, i32 29>
%partial.sums.2 = fadd <32 x float> %partial.sums.4, %partial.partners.2
%partial.partners.1 = shufflevector <32 x float> %partial.sums.2, <32 x float> poison, <32 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14, i32 17, i32 16, i32 19, i32 18, i32 21, i32 20, i32 23, i32 22, i32 25, i32 24, i32 27, i32 26, i32 29, i32 28, i32 31, i32 30>
%partial.sums.1 = fadd <32 x float> %partial.sums.2, %partial.partners.1
%sum = extractelement <32 x float> %partial.sums.1, i32 0
%bias.base = mul i32 %out.channels, %terms
%bias.index = add i32 %bias.base, %channel
%bias.wide.index = zext i32 %bias.index to i64
br i1 %has.bias, label %bias.load, label %bias.zero
bias.load:
%bias.decode.index = add i64 %weight.base.wide, %bias.wide.index
%bias.packed = icmp ne i32 %decode, 0
br i1 %bias.packed, label %bias.packed.load, label %bias.dense.load
bias.dense.load:
%bias.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %bias.wide.index
%bias.dense.model = load i8, ptr addrspace(1) %bias.ptr, align 1
br label %bias.ready
bias.packed.load:
%bias.packed.model = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %bias.decode.index, i32 %decode)
br label %bias.ready
bias.zero:
%bias.zero.model = call i8 @recipe.encode(float %state.zero)
br label %bias.ready
bias.ready:
%bias.model = phi i8 [ %bias.dense.model, %bias.dense.load ], [ %bias.packed.model, %bias.packed.load ], [ %bias.zero.model, %bias.zero ]
%bias.wide = call float @recipe.decode(i8 %bias.model)
%sum.bias = call float @recipe.state.add(float %sum, float %bias.wide)
%sum.value = select i1 %has.bias, float %sum.bias, float %sum
%result.model = call i8 @recipe.encode(float %sum.value)
%result.positive = call i1 @recipe.ogt(i8 %result.model, i8 0)
%result.activated = select i1 %result.positive, i8 %result.model, i8 0
%result = select i1 %relu, i8 %result.activated, i8 %result.model
%output.channel = zext i32 %channel to i64
%out.length.wide = zext i32 %out.length to i64
%output.channel.base = mul i64 %output.channel, %out.length.wide
%output.index = add i64 %output.channel.base, %position
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.index
store i8 %result, ptr addrspace(1) %output.ptr, align 1
br label %job.done
job.done:
%job.next = add i32 %job, %groups
br label %job.loop
exit:
ret void
}
; One output row per wave. The two waves in the normal gfx11 workgroup each
; walk adjacent K elements, then use the wave fadd reduction without LDS.
define internal void @contraction_forward_gemv_wave_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel,
i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode ) #1 { entry:
%exact.partials = alloca [32 x float], align 4
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%groups = udiv i32 %threads, %block
%width = call i32 @recipe.wavefront.width()
%exact.serial = icmp eq i32 %width, 1
%waves = udiv i32 %block, %width
%wave = udiv i32 %lid, %width
%lane = urem i32 %lid, %width
%exact.active = icmp ult i32 %lane, 32
%exact.first = select i1 %exact.active, i32 %lane, i32 32
%state.zero = call float @recipe.state.from.u1(i1 false)
%terms = add i32 %in.channels, 0
%terms.wide = zext i32 %terms to i64
%in.length.wide = zext i32 %in.length to i64
%out.end = add i32 %out.begin, %out.span
%weight.base.wide = add i64 %weight.base, 0
%jobs.adjusted = add i32 %out.channels, %waves
%jobs.numerator = sub i32 %jobs.adjusted, 1
%jobs = udiv i32 %jobs.numerator, %waves
; A stored weight of one or two planes of 256-value blocks, rows of the
; second after the first: a row's format and byte base follow its plane.
%plane.rows = call i32 @recipe.model.plane.rows(i32 %decode)
%plane.base = call i64 @recipe.model.plane.base(i32 %decode)
%plane.kind.0 = call i32 @recipe.model.plane.kind(i32 %decode, i32 0)
%plane.kind.1 = call i32 @recipe.model.plane.kind(i32 %decode, i32 1)
%plane.0.q4 = icmp eq i32 %plane.kind.0, 1
%plane.1.q4 = icmp eq i32 %plane.kind.1, 1
%plane.0.q6 = icmp eq i32 %plane.kind.0, 2
%plane.1.q6 = icmp eq i32 %plane.kind.1, 2
%q4.selector = or i1 %plane.0.q4, %plane.1.q4
%int8.backend = call i1 @recipe.int8.dots()
%int8.declared = call i1 @recipe.model.int.activations(i32 %decode)
%int.width = call i32 @recipe.model.int.width(i32 %decode)
%int16 = icmp eq i32 %int.width, 16
%int32 = icmp eq i32 %int.width, 32
%int.available = or i1 %int8.backend, %int16
%int8.dots = and i1 %int.available, %int8.declared
%q4.remainder = urem i32 %terms, 256
%q4.aligned = icmp eq i32 %q4.remainder, 0
%q4.available = and i1 %q4.selector, %q4.aligned
%q6.selector = or i1 %plane.0.q6, %plane.1.q6
%q6.available = and i1 %q6.selector, %q4.aligned
%b32.kind = call i32 @recipe.model.block32(i32 %decode)
%b32.stride = call i64 @recipe.model.block32.stride(i32 %decode)
%b32.selector = icmp ne i32 %b32.kind, 0
%b32.remainder = urem i32 %terms, 32
%b32.aligned = icmp eq i32 %b32.remainder, 0
%b32.available = and i1 %b32.selector, %b32.aligned
%q8.k = or i1 %q4.available, %q6.available
%block.available = or i1 %q8.k, %b32.available
; An int(n) sum on a backend with int8 dots rounds its activations to int8
; codes for the dot4 helpers, as the block declared; every other sum stages
; its activation column as it is and dots the stored codes exactly.
%exact = xor i1 %int8.dots, true
%q8.available = and i1 %block.available, %int8.dots
%stage.available = and i1 %block.available, %exact
; The column is staged in chunks the tile can hold, whole 256-value blocks at
; a time; a row's partial sums meet in its output between chunks.
%tile.bytes = call i32 @recipe.tile.bytes()
%tile.capacity = udiv i32 %tile.bytes, 1
%tile.blocks = udiv i32 %tile.capacity, 256
%tile.chunk = mul i32 %tile.blocks, 256
; Floating packed sums use whole 256-value blocks and 32 logical partials
; on every backend, independently of staging capacity and physical wave width.
%chunk.span = select i1 %stage.available, i32 256, i32 %terms
%stage.tile = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 0
%q8.shared = getelementptr i8, ptr addrspace(3) @contraction_tile_f8, i64 0
%q8.blocks = udiv i32 %terms, 32
; One int8 activation record per 32 inputs: the step's scale in the state,
; then the 32 codes.
%int.code.bytes = select i1 %int16, i64 2, i64 1
%int.block.bytes = mul i64 32, %int.code.bytes
%q8.record = add i64 %int.block.bytes, 4
%q8.block.record = mul i64 %q8.record, 8
; The inputs that share one step, in 32-value records: the block's int(n)
; step, 32 as the weight blocks or what it declared (256 is llama.cpp's Q8_K).
%q8.step = call i32 @recipe.model.int.step(i32 %decode)
%q8.span = udiv i32 %q8.step, 32
%q8.groups.adjusted = add i32 %q8.blocks, %q8.span
%q8.groups.numerator = sub i32 %q8.groups.adjusted, 1
%q8.groups = udiv i32 %q8.groups.numerator, %q8.span
%q8.width.half = udiv i32 %width, 2
%q8.zero = call float @recipe.state.from.u1(i1 false)
%q8.one = call float @recipe.state.from.u1(i1 true)
%int.levels = select i1 %int16, i32 32767, i32 127
%int.minimum = select i1 %int16, i32 -32768, i32 -128
%q8.levels = call float @recipe.state.from.u32(i32 %int.levels)
%dense.width.wide = icmp ugt i32 %width, 32
%dense.width = select i1 %dense.width.wide, i32 32, i32 %width
%dense.lane.active = icmp ult i32 %lane, %dense.width
%dense.first = select i1 %dense.lane.active, i32 %lane, i32 %terms
br label %position.loop
position.loop:
%position.index = phi i32 [ %out.begin, %entry ], [ %position.next, %position.done ]
%position.more = icmp ult i32 %position.index, %out.end
br i1 %position.more, label %position.step, label %exit
position.step:
%position = zext i32 %position.index to i64
br label %chunk.loop
chunk.loop:
%chunk.base = phi i32 [ 0, %position.step ], [ %chunk.next, %chunk.done ]
%chunk.remaining = sub i32 %terms, %chunk.base
%chunk.over = icmp ugt i32 %chunk.remaining, %chunk.span
%chunk.terms = select i1 %chunk.over, i32 %chunk.span, i32 %chunk.remaining
%chunk.end = add i32 %chunk.base, %chunk.terms
%chunk.first = icmp eq i32 %chunk.base, 0
%chunk.last = icmp eq i32 %chunk.end, %terms
%chunk.pitch = udiv i32 %chunk.terms, 4
%chunk.base.wide = zext i32 %chunk.base to i64
%chunk.k.blocks = udiv i32 %chunk.base, 256
%chunk.b32.blocks = udiv i32 %chunk.base, 32
br i1 %q8.available, label %q8.entry, label %stage.check
stage.check:
br i1 %stage.available, label %stage.entry, label %job.loop
; The activation column of this position, staged once per workgroup in the
; model type, quad-interleaved so a lane reads each quad of its slice as one
; vector and adjacent lanes read adjacent vectors.
stage.entry:
br label %stage.loop
stage.loop:
%stage.c = phi i32 [ %lid, %stage.entry ], [ %stage.c.next, %stage.step ]
%stage.more = icmp ult i32 %stage.c, %chunk.terms
br i1 %stage.more, label %stage.step, label %stage.done
stage.step:
%stage.c.local = zext i32 %stage.c to i64
%stage.c.wide = add i64 %stage.c.local, %chunk.base.wide
%stage.index = mul i64 %stage.c.wide, %in.length.wide
%stage.at = add i64 %stage.index, %position
%stage.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %stage.at
%stage.value = load i8, ptr addrspace(1) %stage.ptr, align 1
%stage.within = and i32 %stage.c, 15
%stage.quad = lshr i32 %stage.within, 2
%stage.lane = and i32 %stage.c, 3
%stage.column = lshr i32 %stage.c, 4
%stage.column.quad = mul i32 %stage.column, 4
%stage.row.base = mul i32 %stage.quad, %chunk.pitch
%stage.at.column = add i32 %stage.row.base, %stage.column.quad
%stage.at.slot = add i32 %stage.at.column, %stage.lane
%stage.slot = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %stage.at.slot
store i8 %stage.value, ptr addrspace(3) %stage.slot, align 1
%stage.c.next = add i32 %stage.c, %block
br label %stage.loop
stage.done:
call void @llvm.nvvm.barrier0()
br label %job.loop
; Quantize each 32-value activation block with one stored step.
q8.entry:
br label %q8.group.loop
q8.group.loop:
%q8.group = phi i32 [ %wave, %q8.entry ], [ %q8.group.next, %q8.group.done ]
%q8.group.more = icmp ult i32 %q8.group, %q8.groups
br i1 %q8.group.more, label %q8.group.step, label %q8.done
q8.group.step:
%q8.group.first = mul i32 %q8.group, %q8.span
%q8.group.stop.raw = add i32 %q8.group.first, %q8.span
%q8.group.over = icmp ult i32 %q8.blocks, %q8.group.stop.raw
%q8.group.stop = select i1 %q8.group.over, i32 %q8.blocks, i32 %q8.group.stop.raw
br label %q8.extreme.loop
q8.extreme.loop:
%q8.extreme.block = phi i32 [ %q8.group.first, %q8.group.step ], [ %q8.extreme.block.next, %q8.extreme.lane.done ]
%q8.extreme = phi float [ %q8.zero, %q8.group.step ], [ %q8.extreme.acc, %q8.extreme.lane.done ]
%q8.extreme.more = icmp ult i32 %q8.extreme.block, %q8.group.stop
br i1 %q8.extreme.more, label %q8.extreme.step, label %q8.extreme.done
; Each lane walks the block's values at the wave width: one value per lane
; in a wave of 32, every value on a width of one.
q8.extreme.step:
%q8.extreme.term.base = mul i32 %q8.extreme.block, 32
br label %q8.extreme.lane.loop
q8.extreme.lane.loop:
%q8.extreme.v = phi i32 [ %lane, %q8.extreme.step ], [ %q8.extreme.v.next, %q8.extreme.lane.step ]
%q8.extreme.acc = phi float [ %q8.extreme, %q8.extreme.step ], [ %q8.extreme.next, %q8.extreme.lane.step ]
%q8.extreme.lane.more = icmp ult i32 %q8.extreme.v, 32
br i1 %q8.extreme.lane.more, label %q8.extreme.lane.step, label %q8.extreme.lane.done
q8.extreme.lane.step:
%q8.extreme.term = add i32 %q8.extreme.term.base, %q8.extreme.v
%q8.extreme.term.wide = zext i32 %q8.extreme.term to i64
%q8.extreme.offset = mul i64 %q8.extreme.term.wide, %in.length.wide
%q8.extreme.index = add i64 %q8.extreme.offset, %position
%q8.extreme.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %q8.extreme.index
%q8.extreme.model = load i8, ptr addrspace(1) %q8.extreme.ptr, align 1
%q8.extreme.value = call float @recipe.decode(i8 %q8.extreme.model)
%q8.extreme.abs = call float @recipe.state.abs(float %q8.extreme.value)
%q8.extreme.have = call float @recipe.state.abs(float %q8.extreme.acc)
%q8.extreme.larger = call i1 @recipe.state.ogt(float %q8.extreme.abs, float %q8.extreme.have)
%q8.extreme.next = select i1 %q8.extreme.larger, float %q8.extreme.value, float %q8.extreme.acc
%q8.extreme.v.next = add i32 %q8.extreme.v, %width
br label %q8.extreme.lane.loop
q8.extreme.lane.done:
%q8.extreme.block.next = add i32 %q8.extreme.block, 1
br label %q8.extreme.loop
q8.extreme.done:
br label %q8.max.loop
q8.max.loop:
%q8.max.offset = phi i32 [ %q8.width.half, %q8.extreme.done ], [ %q8.max.offset.next, %q8.max.step ]
%q8.max.value = phi float [ %q8.extreme, %q8.extreme.done ], [ %q8.max.next, %q8.max.step ]
%q8.max.more = icmp ugt i32 %q8.max.offset, 0
br i1 %q8.max.more, label %q8.max.step, label %q8.max.done
q8.max.step:
%q8.max.partner.lane = xor i32 %lane, %q8.max.offset
%q8.max.partner.index = mul i32 %q8.max.partner.lane, 4
%q8.max.partner = call float @recipe.wave.partner(float %q8.max.value, i32 %q8.max.partner.index)
%q8.max.partner.abs = call float @recipe.state.abs(float %q8.max.partner)
%q8.max.value.abs = call float @recipe.state.abs(float %q8.max.value)
%q8.max.greater = call i1 @recipe.state.ogt(float %q8.max.partner.abs, float %q8.max.value.abs)
%q8.max.equal = fcmp oeq float %q8.max.partner.abs, %q8.max.value.abs
%q8.max.signed = call i1 @recipe.state.ogt(float %q8.max.partner, float %q8.max.value)
%q8.max.tie = and i1 %q8.max.equal, %q8.max.signed
%q8.max.choose = or i1 %q8.max.greater, %q8.max.tie
%q8.max.next = select i1 %q8.max.choose, float %q8.max.partner, float %q8.max.value
%q8.max.offset.next = udiv i32 %q8.max.offset, 2
br label %q8.max.loop
q8.max.done:
%q8.max.abs = call float @recipe.state.abs(float %q8.max.value)
%q8.nonzero = call i1 @recipe.state.ogt(float %q8.max.abs, float %q8.zero)
%q8.max.safe = select i1 %q8.nonzero, float %q8.max.value, float %q8.one
; Match signed scale and reciprocal before rounding codes.
%q8.levels.negative = call float @recipe.state.neg(float %q8.levels)
%q8.inverse.raw = call float @recipe.state.div(float %q8.levels.negative, float %q8.max.safe)
%q8.d.raw = call float @recipe.state.div(float %q8.one, float %q8.inverse.raw)
%q8.inverse = select i1 %q8.nonzero, float %q8.inverse.raw, float %q8.zero
%q8.d = select i1 %q8.nonzero, float %q8.d.raw, float %q8.zero
br label %q8.code.loop
q8.code.loop:
%q8.code.block = phi i32 [ %q8.group.first, %q8.max.done ], [ %q8.code.block.next, %q8.code.lane.done ]
%q8.code.more = icmp ult i32 %q8.code.block, %q8.group.stop
br i1 %q8.code.more, label %q8.code.step, label %q8.group.done
q8.code.step:
%q8.code.term.base = mul i32 %q8.code.block, 32
%q8.code.block.wide = zext i32 %q8.code.block to i64
%q8.code.block.offset = mul i64 %q8.code.block.wide, %q8.record
%q8.code.block.ptr = getelementptr i8, ptr addrspace(3) %q8.shared, i64 %q8.code.block.offset
%q8.code.owner = icmp eq i32 %lane, 0
br i1 %q8.code.owner, label %q8.code.meta, label %q8.code.lane.loop
q8.code.meta:
store float %q8.d, ptr addrspace(3) %q8.code.block.ptr, align 4
br label %q8.code.lane.loop
q8.code.lane.loop:
%q8.code.v = phi i32 [ %lane, %q8.code.step ], [ %lane, %q8.code.meta ], [ %q8.code.v.next, %q8.code.stored ]
%q8.code.lane.more = icmp ult i32 %q8.code.v, 32
br i1 %q8.code.lane.more, label %q8.code.lane.step, label %q8.code.lane.done
q8.code.lane.step:
%q8.code.term = add i32 %q8.code.term.base, %q8.code.v
%q8.code.term.wide = zext i32 %q8.code.term to i64
%q8.code.offset = mul i64 %q8.code.term.wide, %in.length.wide
%q8.code.index = add i64 %q8.code.offset, %position
%q8.code.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %q8.code.index
%q8.code.model = load i8, ptr addrspace(1) %q8.code.ptr, align 1
%q8.code.value = call float @recipe.decode(i8 %q8.code.model)
%q8.code.scaled = call float @recipe.state.mul(float %q8.code.value, float %q8.inverse)
%q8.code.even = call float @recipe.state.roundeven(float %q8.code.scaled)
%q8.code.int.raw = call i32 @recipe.state.to.s32(float %q8.code.even)
%q8.code.low = icmp slt i32 %q8.code.int.raw, %int.minimum
%q8.code.low.clamped = select i1 %q8.code.low, i32 %int.minimum, i32 %q8.code.int.raw
%q8.code.high = icmp sgt i32 %q8.code.low.clamped, %int.levels
%q8.code.int = select i1 %q8.code.high, i32 %int.levels, i32 %q8.code.low.clamped
br label %q8.code.store
q8.code.store:
%q8.code.v.wide = zext i32 %q8.code.v to i64
%q8.code.ptr.base = getelementptr i8, ptr addrspace(3) %q8.code.block.ptr, i64 4
%q8.code.byte.offset = mul i64 %q8.code.v.wide, %int.code.bytes
%q8.code.byte.ptr = getelementptr i8, ptr addrspace(3) %q8.code.ptr.base, i64 %q8.code.byte.offset
br i1 %int16, label %q8.code.store16, label %q8.code.store8
q8.code.store16:
%q8.code.half = trunc i32 %q8.code.int to i16
store i16 %q8.code.half, ptr addrspace(3) %q8.code.byte.ptr, align 2
br label %q8.code.stored
q8.code.store8:
%q8.code.byte = trunc i32 %q8.code.int to i8
store i8 %q8.code.byte, ptr addrspace(3) %q8.code.byte.ptr, align 1
br label %q8.code.stored
q8.code.stored:
%q8.code.v.next = add i32 %q8.code.v, %width
br label %q8.code.lane.loop
q8.code.lane.done:
%q8.code.block.next = add i32 %q8.code.block, 1
br label %q8.code.loop
q8.group.done:
%q8.group.next = add i32 %q8.group, %waves
br label %q8.group.loop
q8.done:
call void @llvm.nvvm.barrier0()
br label %job.loop
job.loop:
%job = phi i32 [ %group, %stage.check ], [ %group, %stage.done ], [ %group, %q8.done ], [ %job.next, %job.done ]
%job.more = icmp ult i32 %job, %jobs
br i1 %job.more, label %job.step, label %chunk.done
job.step:
%channel.base = mul i32 %job, %waves
%channel = add i32 %channel.base, %wave
%channel.active = icmp ult i32 %channel, %out.channels
%channel.safe = select i1 %channel.active, i32 %channel, i32 0
%channel.wide = zext i32 %channel.safe to i64
%channel.offset = mul i64 %channel.wide, %terms.wide
%row.second = icmp uge i32 %channel.safe, %plane.rows
%row.kind = select i1 %row.second, i32 %plane.kind.1, i32 %plane.kind.0
%row.local.raw = sub i32 %channel.safe, %plane.rows
%row.local = select i1 %row.second, i32 %row.local.raw, i32 %channel.safe
%row.local.wide = zext i32 %row.local to i64
%row.q4 = icmp eq i32 %row.kind, 1
%row.q6 = icmp eq i32 %row.kind, 2
%row.stride = select i1 %row.q4, i64 144, i64 210
%row.blocks = udiv i32 %terms, 256
%row.blocks.wide = zext i32 %row.blocks to i64
%row.bytes = mul i64 %row.blocks.wide, %row.stride
%row.offset = mul i64 %row.local.wide, %row.bytes
%row.plane.base = select i1 %row.second, i64 %plane.base, i64 0
%row.base = add i64 %row.plane.base, %row.offset
%row.q4.on = and i1 %q4.available, %row.q4
%row.q6.on = and i1 %q6.available, %row.q6
; A Q4_K row under a 256-value step sums each block as llama.cpp's Q4_K
; kernels do: the block's codes times scales exactly in i32, then one fma
; per block in block order, the minimums on their own chain, subtracted last.
%q8.is256 = icmp eq i32 %q8.span, 8
%row.q4.block = and i1 %row.q4.on, %q8.is256
br i1 %exact, label %exact.partition, label %q4.check
exact.partition:
%exact.lane = phi i32 [ %exact.first, %job.step ], [ %exact.lane.next, %exact.serial.store ]
br label %exact.q4.check
q4.check:
br i1 %row.q4.block, label %q4b.loop, label %q4.check.slices
q4.check.slices:
br i1 %row.q4.on, label %q4.sum.loop, label %q6.check
q4b.loop:
%q4b.block = phi i32 [ 0, %q4.check ], [ %q4b.block.next, %q4b.done ]
%q4b.acc = phi float [ %state.zero, %q4.check ], [ %q4b.acc.next, %q4b.done ]
%q4b.accmin = phi float [ %state.zero, %q4.check ], [ %q4b.accmin.next, %q4b.done ]
%q4b.blocks = udiv i32 %terms, 256
%q4b.more = icmp ult i32 %q4b.block, %q4b.blocks
br i1 %q4b.more, label %q4b.step, label %q4b.exit
q4b.step:
%q4b.block.wide = zext i32 %q4b.block to i64
%q4b.block.bytes = mul i64 %q4b.block.wide, 144
%q4b.byte = add i64 %row.base, %q4b.block.bytes
%q4b.q8.offset = mul i64 %q4b.block.wide, %q8.block.record
%q4b.q8 = getelementptr i8, ptr addrspace(3) %q8.shared, i64 %q4b.q8.offset
br label %q4b.slice.loop
q4b.slice.loop:
; Sixteen slices of sixteen values; llama.cpp folds each quarter of the block
; (two sub-blocks, 64 values) into one exact int and one fma, so the slices
; keep four int sums, one per quarter, and the quarters meet the accumulator
; in order. A quarter's int stays under 2^24, so its float is exact.
%q4b.s = phi i32 [ %lane, %q4b.step ], [ %q4b.s.next, %q4b.slice.step ]
%q4b.dot0 = phi i32 [ 0, %q4b.step ], [ %q4b.dot0.next, %q4b.slice.step ]
%q4b.dot1 = phi i32 [ 0, %q4b.step ], [ %q4b.dot1.next, %q4b.slice.step ]
%q4b.dot2 = phi i32 [ 0, %q4b.step ], [ %q4b.dot2.next, %q4b.slice.step ]
%q4b.dot3 = phi i32 [ 0, %q4b.step ], [ %q4b.dot3.next, %q4b.slice.step ]
%q4b.min0 = phi i32 [ 0, %q4b.step ], [ %q4b.min0.next, %q4b.slice.step ]
%q4b.min1 = phi i32 [ 0, %q4b.step ], [ %q4b.min1.next, %q4b.slice.step ]
%q4b.min2 = phi i32 [ 0, %q4b.step ], [ %q4b.min2.next, %q4b.slice.step ]
%q4b.min3 = phi i32 [ 0, %q4b.step ], [ %q4b.min3.next, %q4b.slice.step ]
%q4b.s.more = icmp ult i32 %q4b.s, 16
br i1 %q4b.s.more, label %q4b.slice.step, label %q4b.slice.done
q4b.slice.step:
%q4b.ints = call i64 @recipe.q4k.ints(ptr addrspace(1) %weights, i64 %q4b.byte, ptr addrspace(3) %q4b.q8, i32 %q4b.s)
%q4b.ints.dot.wide = ashr i64 %q4b.ints, 32
%q4b.ints.dot = trunc i64 %q4b.ints.dot.wide to i32
%q4b.ints.min = trunc i64 %q4b.ints to i32
%q4b.quarter = lshr i32 %q4b.s, 2
%q4b.in0 = icmp eq i32 %q4b.quarter, 0
%q4b.in1 = icmp eq i32 %q4b.quarter, 1
%q4b.in2 = icmp eq i32 %q4b.quarter, 2
%q4b.in3 = icmp eq i32 %q4b.quarter, 3
%q4b.dot0.add = select i1 %q4b.in0, i32 %q4b.ints.dot, i32 0
%q4b.dot1.add = select i1 %q4b.in1, i32 %q4b.ints.dot, i32 0
%q4b.dot2.add = select i1 %q4b.in2, i32 %q4b.ints.dot, i32 0
%q4b.dot3.add = select i1 %q4b.in3, i32 %q4b.ints.dot, i32 0
%q4b.min0.add = select i1 %q4b.in0, i32 %q4b.ints.min, i32 0
%q4b.min1.add = select i1 %q4b.in1, i32 %q4b.ints.min, i32 0
%q4b.min2.add = select i1 %q4b.in2, i32 %q4b.ints.min, i32 0
%q4b.min3.add = select i1 %q4b.in3, i32 %q4b.ints.min, i32 0
%q4b.dot0.next = add i32 %q4b.dot0, %q4b.dot0.add
%q4b.dot1.next = add i32 %q4b.dot1, %q4b.dot1.add
%q4b.dot2.next = add i32 %q4b.dot2, %q4b.dot2.add
%q4b.dot3.next = add i32 %q4b.dot3, %q4b.dot3.add
%q4b.min0.next = add i32 %q4b.min0, %q4b.min0.add
%q4b.min1.next = add i32 %q4b.min1, %q4b.min1.add
%q4b.min2.next = add i32 %q4b.min2, %q4b.min2.add
%q4b.min3.next = add i32 %q4b.min3, %q4b.min3.add
%q4b.s.next = add i32 %q4b.s, %width
br label %q4b.slice.loop
q4b.slice.done:
; The lanes that hold slices meet: an int sum per quarter, exact in any
; order; lanes past the sixteenth hold zero, and every lane ends with all four.
%q4b.red.width.raw = icmp ult i32 %width, 16
%q4b.red.width = select i1 %q4b.red.width.raw, i32 %width, i32 16
%q4b.red.initial = lshr i32 %q4b.red.width, 1
br label %q4b.red.loop
q4b.red.loop:
%q4b.red.offset = phi i32 [ %q4b.red.initial, %q4b.slice.done ], [ %q4b.red.offset.next, %q4b.red.step ]
%q4b.red.dot0 = phi i32 [ %q4b.dot0, %q4b.slice.done ], [ %q4b.red.dot0.next, %q4b.red.step ]
%q4b.red.dot1 = phi i32 [ %q4b.dot1, %q4b.slice.done ], [ %q4b.red.dot1.next, %q4b.red.step ]
%q4b.red.dot2 = phi i32 [ %q4b.dot2, %q4b.slice.done ], [ %q4b.red.dot2.next, %q4b.red.step ]
%q4b.red.dot3 = phi i32 [ %q4b.dot3, %q4b.slice.done ], [ %q4b.red.dot3.next, %q4b.red.step ]
%q4b.red.min0 = phi i32 [ %q4b.min0, %q4b.slice.done ], [ %q4b.red.min0.next, %q4b.red.step ]
%q4b.red.min1 = phi i32 [ %q4b.min1, %q4b.slice.done ], [ %q4b.red.min1.next, %q4b.red.step ]
%q4b.red.min2 = phi i32 [ %q4b.min2, %q4b.slice.done ], [ %q4b.red.min2.next, %q4b.red.step ]
%q4b.red.min3 = phi i32 [ %q4b.min3, %q4b.slice.done ], [ %q4b.red.min3.next, %q4b.red.step ]
%q4b.red.more = icmp ugt i32 %q4b.red.offset, 0
br i1 %q4b.red.more, label %q4b.red.step, label %q4b.red.done
q4b.red.step:
%q4b.red.partner.lane = xor i32 %lane, %q4b.red.offset
%q4b.red.partner.index = mul i32 %q4b.red.partner.lane, 4
%q4b.red.dot0.bits = bitcast i32 %q4b.red.dot0 to float
%q4b.red.dot0.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.dot0.bits, i32 %q4b.red.partner.index)
%q4b.red.dot0.partner = bitcast float %q4b.red.dot0.partner.bits to i32
%q4b.red.dot0.next = add i32 %q4b.red.dot0, %q4b.red.dot0.partner
%q4b.red.min0.bits = bitcast i32 %q4b.red.min0 to float
%q4b.red.min0.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.min0.bits, i32 %q4b.red.partner.index)
%q4b.red.min0.partner = bitcast float %q4b.red.min0.partner.bits to i32
%q4b.red.min0.next = add i32 %q4b.red.min0, %q4b.red.min0.partner
%q4b.red.dot1.bits = bitcast i32 %q4b.red.dot1 to float
%q4b.red.dot1.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.dot1.bits, i32 %q4b.red.partner.index)
%q4b.red.dot1.partner = bitcast float %q4b.red.dot1.partner.bits to i32
%q4b.red.dot1.next = add i32 %q4b.red.dot1, %q4b.red.dot1.partner
%q4b.red.min1.bits = bitcast i32 %q4b.red.min1 to float
%q4b.red.min1.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.min1.bits, i32 %q4b.red.partner.index)
%q4b.red.min1.partner = bitcast float %q4b.red.min1.partner.bits to i32
%q4b.red.min1.next = add i32 %q4b.red.min1, %q4b.red.min1.partner
%q4b.red.dot2.bits = bitcast i32 %q4b.red.dot2 to float
%q4b.red.dot2.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.dot2.bits, i32 %q4b.red.partner.index)
%q4b.red.dot2.partner = bitcast float %q4b.red.dot2.partner.bits to i32
%q4b.red.dot2.next = add i32 %q4b.red.dot2, %q4b.red.dot2.partner
%q4b.red.min2.bits = bitcast i32 %q4b.red.min2 to float
%q4b.red.min2.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.min2.bits, i32 %q4b.red.partner.index)
%q4b.red.min2.partner = bitcast float %q4b.red.min2.partner.bits to i32
%q4b.red.min2.next = add i32 %q4b.red.min2, %q4b.red.min2.partner
%q4b.red.dot3.bits = bitcast i32 %q4b.red.dot3 to float
%q4b.red.dot3.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.dot3.bits, i32 %q4b.red.partner.index)
%q4b.red.dot3.partner = bitcast float %q4b.red.dot3.partner.bits to i32
%q4b.red.dot3.next = add i32 %q4b.red.dot3, %q4b.red.dot3.partner
%q4b.red.min3.bits = bitcast i32 %q4b.red.min3 to float
%q4b.red.min3.partner.bits = call float @recipe.wave.partner.f32(float %q4b.red.min3.bits, i32 %q4b.red.partner.index)
%q4b.red.min3.partner = bitcast float %q4b.red.min3.partner.bits to i32
%q4b.red.min3.next = add i32 %q4b.red.min3, %q4b.red.min3.partner
%q4b.red.offset.next = lshr i32 %q4b.red.offset, 1
br label %q4b.red.loop
q4b.red.done:
%q4b.d.ptr = getelementptr i8, ptr addrspace(1) %weights, i64 %q4b.byte
%q4b.d.bits = load half, ptr addrspace(1) %q4b.d.ptr, align 2
%q4b.d = call float @recipe.state.from.f16(half %q4b.d.bits)
%q4b.dmin.ptr = getelementptr i8, ptr addrspace(1) %q4b.d.ptr, i64 2
%q4b.dmin.bits = load half, ptr addrspace(1) %q4b.dmin.ptr, align 2
%q4b.dmin = call float @recipe.state.from.f16(half %q4b.dmin.bits)
%q4b.d8 = load float, ptr addrspace(3) %q4b.q8, align 4
%q4b.scale = call float @recipe.state.mul(float %q4b.d, float %q4b.d8)
%q4b.dscale = call float @recipe.state.mul(float %q4b.dmin, float %q4b.d8)
%q4b.dot0.value = call float @recipe.state.from.s32(i32 %q4b.red.dot0)
%q4b.min0.value = call float @recipe.state.from.s32(i32 %q4b.red.min0)
%q4b.acc.q0 = call float @recipe.state.madd(float %q4b.acc, float %q4b.dot0.value, float %q4b.scale)
%q4b.accmin.q0 = call float @recipe.state.madd(float %q4b.accmin, float %q4b.min0.value, float %q4b.dscale)
%q4b.dot1.value = call float @recipe.state.from.s32(i32 %q4b.red.dot1)
%q4b.min1.value = call float @recipe.state.from.s32(i32 %q4b.red.min1)
%q4b.acc.q1 = call float @recipe.state.madd(float %q4b.acc.q0, float %q4b.dot1.value, float %q4b.scale)
%q4b.accmin.q1 = call float @recipe.state.madd(float %q4b.accmin.q0, float %q4b.min1.value, float %q4b.dscale)
%q4b.dot2.value = call float @recipe.state.from.s32(i32 %q4b.red.dot2)
%q4b.min2.value = call float @recipe.state.from.s32(i32 %q4b.red.min2)
%q4b.acc.q2 = call float @recipe.state.madd(float %q4b.acc.q1, float %q4b.dot2.value, float %q4b.scale)
%q4b.accmin.q2 = call float @recipe.state.madd(float %q4b.accmin.q1, float %q4b.min2.value, float %q4b.dscale)
%q4b.dot3.value = call float @recipe.state.from.s32(i32 %q4b.red.dot3)
%q4b.min3.value = call float @recipe.state.from.s32(i32 %q4b.red.min3)
%q4b.acc.q3 = call float @recipe.state.madd(float %q4b.acc.q2, float %q4b.dot3.value, float %q4b.scale)
%q4b.accmin.q3 = call float @recipe.state.madd(float %q4b.accmin.q2, float %q4b.min3.value, float %q4b.dscale)
; llama.cpp feeds a batch's first 4*(N/4) rows through its gemm, four fmas
; per block as above, and the rows past them (and every single-token step)
; through its gemv, the whole block's int in one fma; a row follows the
; kernel its position would meet there. After the last attention llama.cpp
; keeps only the batch's last row, which its gemv folds whole.
%q4b.dot01 = add i32 %q4b.red.dot0, %q4b.red.dot1
%q4b.dot23 = add i32 %q4b.red.dot2, %q4b.red.dot3
%q4b.dot.all = add i32 %q4b.dot01, %q4b.dot23
%q4b.min01 = add i32 %q4b.red.min0, %q4b.red.min1
%q4b.min23 = add i32 %q4b.red.min2, %q4b.red.min3
%q4b.min.all = add i32 %q4b.min01, %q4b.min23
%q4b.dot.all.value = call float @recipe.state.from.s32(i32 %q4b.dot.all)
%q4b.min.all.value = call float @recipe.state.from.s32(i32 %q4b.min.all)
%q4b.acc.whole = call float @recipe.state.madd(float %q4b.acc, float %q4b.dot.all.value, float %q4b.scale)
%q4b.accmin.whole = call float @recipe.state.madd(float %q4b.accmin, float %q4b.min.all.value, float %q4b.dscale)
%q4b.batch.local = sub i32 %position.index, %out.begin
%q4b.batch.rem = and i32 %out.span, 3
%q4b.gemm.end = sub i32 %out.span, %q4b.batch.rem
%q4b.whole.rows = icmp uge i32 %q4b.batch.local, %q4b.gemm.end
%q4b.tail = call i1 @recipe.model.tail(i32 %decode)
%q4b.batch.last = sub i32 %out.span, 1
%q4b.is.last = icmp eq i32 %q4b.batch.local, %q4b.batch.last
%q4b.tail.last = and i1 %q4b.tail, %q4b.is.last
%q4b.whole = or i1 %q4b.whole.rows, %q4b.tail.last
%q4b.acc.next = select i1 %q4b.whole, float %q4b.acc.whole, float %q4b.acc.q3
%q4b.accmin.next = select i1 %q4b.whole, float %q4b.accmin.whole, float %q4b.accmin.q3
br label %q4b.done
q4b.done:
%q4b.block.next = add i32 %q4b.block, 1
br label %q4b.loop
q4b.exit:
%q4b.difference = call float @recipe.state.sub(float %q4b.acc, float %q4b.accmin)
%q4b.owner = icmp eq i32 %lane, 0
%q4b.owned = select i1 %q4b.owner, float %q4b.difference, float %state.zero
%q4b.result = select i1 %channel.active, float %q4b.owned, float %state.zero
br label %sum.done
exact.q4.check:
br i1 %row.q4.on, label %exact.q4.loop, label %exact.q6.check
exact.q4.loop:
%exact.q4.slice = phi i32 [ %exact.lane, %exact.q4.check ], [ %exact.q4.slice.next, %exact.q4.step ]
%exact.q4.sum = phi float [ %state.zero, %exact.q4.check ], [ %exact.q4.sum.next, %exact.q4.step ]
%exact.q4.slices = udiv i32 %chunk.terms, 16
%exact.q4.more = icmp ult i32 %exact.q4.slice, %exact.q4.slices
br i1 %exact.q4.more, label %exact.q4.step, label %exact.q4.done
exact.q4.step:
%exact.q4.block.local = udiv i32 %exact.q4.slice, 16
%exact.q4.block = add i32 %exact.q4.block.local, %chunk.k.blocks
%exact.q4.slice.local = urem i32 %exact.q4.slice, 16
%exact.q4.block.wide = zext i32 %exact.q4.block to i64
%exact.q4.block.bytes = mul i64 %exact.q4.block.wide, 144
%exact.q4.byte.offset = add i64 %row.base, %exact.q4.block.bytes
%exact.q4.loaded = call float @recipe.q4k.exact(ptr addrspace(1) %weights, i64 %exact.q4.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.q4.slice, i32 %chunk.pitch, i32 %exact.q4.slice.local)
%exact.q4.value = select i1 %channel.active, float %exact.q4.loaded, float %state.zero
%exact.q4.sum.next = call float @recipe.state.add(float %exact.q4.sum, float %exact.q4.value)
%exact.q4.slice.next = add i32 %exact.q4.slice, 32
br label %exact.q4.loop
exact.q4.done:
br label %sum.done
exact.q6.check:
br i1 %row.q6.on, label %exact.q6.loop, label %exact.b32.check
exact.q6.loop:
%exact.q6.slice = phi i32 [ %exact.lane, %exact.q6.check ], [ %exact.q6.slice.next, %exact.q6.step ]
%exact.q6.sum = phi float [ %state.zero, %exact.q6.check ], [ %exact.q6.sum.next, %exact.q6.step ]
%exact.q6.slices = udiv i32 %chunk.terms, 16
%exact.q6.more = icmp ult i32 %exact.q6.slice, %exact.q6.slices
br i1 %exact.q6.more, label %exact.q6.step, label %exact.q6.done
exact.q6.step:
%exact.q6.block.local = udiv i32 %exact.q6.slice, 16
%exact.q6.block = add i32 %exact.q6.block.local, %chunk.k.blocks
%exact.q6.slice.local = urem i32 %exact.q6.slice, 16
%exact.q6.block.wide = zext i32 %exact.q6.block to i64
%exact.q6.block.bytes = mul i64 %exact.q6.block.wide, 210
%exact.q6.byte.offset = add i64 %row.base, %exact.q6.block.bytes
%exact.q6.loaded = call float @recipe.q6k.exact(ptr addrspace(1) %weights, i64 %exact.q6.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.q6.slice, i32 %chunk.pitch, i32 %exact.q6.slice.local)
%exact.q6.value = select i1 %channel.active, float %exact.q6.loaded, float %state.zero
%exact.q6.sum.next = call float @recipe.state.add(float %exact.q6.sum, float %exact.q6.value)
%exact.q6.slice.next = add i32 %exact.q6.slice, 32
br label %exact.q6.loop
exact.q6.done:
br label %sum.done
exact.b32.check:
br i1 %b32.available, label %exact.b32.loop, label %sum.loop
exact.b32.loop:
%exact.b32.slice = phi i32 [ %exact.lane, %exact.b32.check ], [ %exact.b32.slice.next, %exact.b32.loaded.done ]
%exact.b32.sum = phi float [ %state.zero, %exact.b32.check ], [ %exact.b32.sum.next, %exact.b32.loaded.done ]
%exact.b32.slice.width = select i1 %int32, i32 32, i32 16
%exact.b32.slices = udiv i32 %chunk.terms, %exact.b32.slice.width
%exact.b32.more = icmp ult i32 %exact.b32.slice, %exact.b32.slices
br i1 %exact.b32.more, label %exact.b32.step, label %exact.b32.done
exact.b32.step:
%exact.b32.row.blocks = udiv i32 %terms, 32
%exact.b32.row.blocks.wide = zext i32 %exact.b32.row.blocks to i64
%exact.b32.channel.row = mul i64 %channel.wide, %exact.b32.row.blocks.wide
%exact.b32.block.half = udiv i32 %exact.b32.slice, 2
%exact.b32.block.local = select i1 %int32, i32 %exact.b32.slice, i32 %exact.b32.block.half
%exact.b32.block = add i32 %exact.b32.block.local, %chunk.b32.blocks
%exact.b32.slice.local = urem i32 %exact.b32.slice, 2
%exact.b32.block.wide = zext i32 %exact.b32.block to i64
%exact.b32.block.index = add i64 %exact.b32.channel.row, %exact.b32.block.wide
%exact.b32.byte.offset = mul i64 %exact.b32.block.index, %b32.stride
br i1 %int32, label %exact.b32.load32, label %exact.b32.load16
exact.b32.load32:
%exact.b32.value32 = call float @recipe.block32.fp32(i32 %b32.kind, ptr addrspace(1) %weights, i64 %exact.b32.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.b32.slice, i32 %chunk.pitch)
br label %exact.b32.loaded.done
exact.b32.load16:
%exact.b32.value16 = call float @recipe.block32.exact(i32 %b32.kind, ptr addrspace(1) %weights, i64 %exact.b32.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.b32.slice, i32 %chunk.pitch, i32 %exact.b32.slice.local)
br label %exact.b32.loaded.done
exact.b32.loaded.done:
%exact.b32.loaded = phi float [ %exact.b32.value32, %exact.b32.load32 ], [ %exact.b32.value16, %exact.b32.load16 ]
%exact.b32.value = select i1 %channel.active, float %exact.b32.loaded, float %state.zero
%exact.b32.sum.next = call float @recipe.state.add(float %exact.b32.sum, float %exact.b32.value)
%exact.b32.slice.next = add i32 %exact.b32.slice, 32
br label %exact.b32.loop
exact.b32.done:
br label %sum.done
q4.sum.loop:
%q4.slice = phi i32 [ %lane, %q4.check.slices ], [ %q4.slice.next, %q4.slice.ready ]
%q4.sum = phi float [ %state.zero, %q4.check.slices ], [ %q4.sum.next, %q4.slice.ready ]
%q4.slices = udiv i32 %terms, 16
%q4.slice.more = icmp ult i32 %q4.slice, %q4.slices
br i1 %q4.slice.more, label %q4.sum.step, label %q4.sum.done
q4.sum.step:
%q4.block = udiv i32 %q4.slice, 16
%q4.slice.local = urem i32 %q4.slice, 16
%q4.block.wide = zext i32 %q4.block to i64
%q4.block.bytes = mul i64 %q4.block.wide, 144
%q4.byte.offset = add i64 %row.base, %q4.block.bytes
%q4.q8.offset = mul i64 %q4.block.wide, %q8.block.record
%q4.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.shared, i64 %q4.q8.offset
%q4.loaded = call float @recipe.q4k.slice(ptr addrspace(1) %weights, i64 %q4.byte.offset, ptr addrspace(3) %q4.q8.ptr, i32 %q4.slice.local)
%q4.value.active = select i1 %channel.active, float %q4.loaded, float %state.zero
%q4.sum.next = call float @recipe.state.add(float %q4.sum, float %q4.value.active)
%q4.slice.next = add i32 %q4.slice, %width
br label %q4.slice.ready
q4.slice.ready:
br label %q4.sum.loop
q4.sum.done:
br label %sum.done
q6.check:
; A Q6_K row under a 256-value step sums as llama.cpp's Q6_K kernel does:
; eight int lanes per block, lane l holding the four-value words at 4l of
; every 32-value group less 32 times group l's scaled Q8 sums, each lane
; meeting its own float accumulator once per block, the eight accumulators
; adding in a fixed tree at the end.
%row.q6.block = and i1 %row.q6.on, %q8.is256
br i1 %row.q6.block, label %q6b.loop, label %q6.check.slices
q6.check.slices:
br i1 %row.q6.on, label %q6.sum.loop, label %b32.check
q6b.loop:
%q6b.block = phi i32 [ 0, %q6.check ], [ %q6b.block.next, %q6b.done ]
%q6b.f0 = phi float [ %state.zero, %q6.check ], [ %q6b.f0.next, %q6b.done ]
%q6b.f1 = phi float [ %state.zero, %q6.check ], [ %q6b.f1.next, %q6b.done ]
%q6b.f2 = phi float [ %state.zero, %q6.check ], [ %q6b.f2.next, %q6b.done ]
%q6b.f3 = phi float [ %state.zero, %q6.check ], [ %q6b.f3.next, %q6b.done ]
%q6b.f4 = phi float [ %state.zero, %q6.check ], [ %q6b.f4.next, %q6b.done ]
%q6b.f5 = phi float [ %state.zero, %q6.check ], [ %q6b.f5.next, %q6b.done ]
%q6b.f6 = phi float [ %state.zero, %q6.check ], [ %q6b.f6.next, %q6b.done ]
%q6b.f7 = phi float [ %state.zero, %q6.check ], [ %q6b.f7.next, %q6b.done ]
%q6b.blocks = udiv i32 %terms, 256
%q6b.more = icmp ult i32 %q6b.block, %q6b.blocks
br i1 %q6b.more, label %q6b.step, label %q6b.exit
q6b.step:
%q6b.block.wide = zext i32 %q6b.block to i64
%q6b.block.bytes = mul i64 %q6b.block.wide, 210
%q6b.byte = add i64 %row.base, %q6b.block.bytes
%q6b.q8.offset = mul i64 %q6b.block.wide, %q8.block.record
%q6b.q8 = getelementptr i8, ptr addrspace(3) %q8.shared, i64 %q6b.q8.offset
br label %q6b.role.loop
q6b.role.loop:
; Sixteen roles across the lanes: role l under 8 gathers lane l's words,
; role l over 8 gathers group l's scaled Q8 sums.
%q6b.p = phi i32 [ %lane, %q6b.step ], [ %q6b.p.next, %q6b.call.done ]
%q6b.i0 = phi i32 [ 0, %q6b.step ], [ %q6b.i0.next, %q6b.call.done ]
%q6b.i1 = phi i32 [ 0, %q6b.step ], [ %q6b.i1.next, %q6b.call.done ]
%q6b.i2 = phi i32 [ 0, %q6b.step ], [ %q6b.i2.next, %q6b.call.done ]
%q6b.i3 = phi i32 [ 0, %q6b.step ], [ %q6b.i3.next, %q6b.call.done ]
%q6b.i4 = phi i32 [ 0, %q6b.step ], [ %q6b.i4.next, %q6b.call.done ]
%q6b.i5 = phi i32 [ 0, %q6b.step ], [ %q6b.i5.next, %q6b.call.done ]
%q6b.i6 = phi i32 [ 0, %q6b.step ], [ %q6b.i6.next, %q6b.call.done ]
%q6b.i7 = phi i32 [ 0, %q6b.step ], [ %q6b.i7.next, %q6b.call.done ]
%q6b.p.more = icmp ult i32 %q6b.p, 16
br i1 %q6b.p.more, label %q6b.role.step, label %q6b.role.done
q6b.role.step:
%q6b.l = and i32 %q6b.p, 7
%q6b.role = lshr i32 %q6b.p, 3
%q6b.is.sub = icmp eq i32 %q6b.role, 1
%q6b.l.high = lshr i32 %q6b.l, 2
%q6b.l.word = and i32 %q6b.l, 3
%q6b.l.twice = shl i32 %q6b.l, 1
br label %q6b.call.loop
q6b.call.loop:
%q6b.k = phi i32 [ 0, %q6b.role.step ], [ %q6b.k.next, %q6b.call.step ]
%q6b.gathered = phi i32 [ 0, %q6b.role.step ], [ %q6b.gathered.next, %q6b.call.step ]
%q6b.k.more = icmp ult i32 %q6b.k, 8
br i1 %q6b.k.more, label %q6b.call.step, label %q6b.call.done
q6b.call.step:
%q6b.k.twice = shl i32 %q6b.k, 1
%q6b.slice.words = add i32 %q6b.k.twice, %q6b.l.high
%q6b.k.high = lshr i32 %q6b.k, 2
%q6b.k.word = and i32 %q6b.k, 3
%q6b.slice.sums = add i32 %q6b.l.twice, %q6b.k.high
%q6b.slice = select i1 %q6b.is.sub, i32 %q6b.slice.sums, i32 %q6b.slice.words
%q6b.word = select i1 %q6b.is.sub, i32 %q6b.k.word, i32 %q6b.l.word
%q6b.part = call i64 @recipe.q6k.part(ptr addrspace(1) %weights, i64 %q6b.byte, ptr addrspace(3) %q6b.q8, i32 %q6b.slice, i32 %q6b.word)
%q6b.part.dot.wide = ashr i64 %q6b.part, 32
%q6b.part.dot = trunc i64 %q6b.part.dot.wide to i32
%q6b.part.sum = trunc i64 %q6b.part to i32
%q6b.part.sub = mul i32 %q6b.part.sum, -32
%q6b.term = select i1 %q6b.is.sub, i32 %q6b.part.sub, i32 %q6b.part.dot
%q6b.gathered.next = add i32 %q6b.gathered, %q6b.term
%q6b.k.next = add i32 %q6b.k, 1
br label %q6b.call.loop
q6b.call.done:
%q6b.own0 = icmp eq i32 %q6b.l, 0
%q6b.i0.add = select i1 %q6b.own0, i32 %q6b.gathered, i32 0
%q6b.i0.next = add i32 %q6b.i0, %q6b.i0.add
%q6b.own1 = icmp eq i32 %q6b.l, 1
%q6b.i1.add = select i1 %q6b.own1, i32 %q6b.gathered, i32 0
%q6b.i1.next = add i32 %q6b.i1, %q6b.i1.add
%q6b.own2 = icmp eq i32 %q6b.l, 2
%q6b.i2.add = select i1 %q6b.own2, i32 %q6b.gathered, i32 0
%q6b.i2.next = add i32 %q6b.i2, %q6b.i2.add
%q6b.own3 = icmp eq i32 %q6b.l, 3
%q6b.i3.add = select i1 %q6b.own3, i32 %q6b.gathered, i32 0
%q6b.i3.next = add i32 %q6b.i3, %q6b.i3.add
%q6b.own4 = icmp eq i32 %q6b.l, 4
%q6b.i4.add = select i1 %q6b.own4, i32 %q6b.gathered, i32 0
%q6b.i4.next = add i32 %q6b.i4, %q6b.i4.add
%q6b.own5 = icmp eq i32 %q6b.l, 5
%q6b.i5.add = select i1 %q6b.own5, i32 %q6b.gathered, i32 0
%q6b.i5.next = add i32 %q6b.i5, %q6b.i5.add
%q6b.own6 = icmp eq i32 %q6b.l, 6
%q6b.i6.add = select i1 %q6b.own6, i32 %q6b.gathered, i32 0
%q6b.i6.next = add i32 %q6b.i6, %q6b.i6.add
%q6b.own7 = icmp eq i32 %q6b.l, 7
%q6b.i7.add = select i1 %q6b.own7, i32 %q6b.gathered, i32 0
%q6b.i7.next = add i32 %q6b.i7, %q6b.i7.add
%q6b.p.next = add i32 %q6b.p, %width
br label %q6b.role.loop
q6b.role.done:
; The lanes that took roles meet: an int sum per lane, exact in any order;
; lanes past the sixteenth hold zero, and every lane ends with all eight.
%q6b.red.width.raw = icmp ult i32 %width, 16
%q6b.red.width = select i1 %q6b.red.width.raw, i32 %width, i32 16
%q6b.red.initial = lshr i32 %q6b.red.width, 1
br label %q6b.red.loop
q6b.red.loop:
%q6b.red.offset = phi i32 [ %q6b.red.initial, %q6b.role.done ], [ %q6b.red.offset.next, %q6b.red.step ]
%q6b.red.i0 = phi i32 [ %q6b.i0, %q6b.role.done ], [ %q6b.red.i0.next, %q6b.red.step ]
%q6b.red.i1 = phi i32 [ %q6b.i1, %q6b.role.done ], [ %q6b.red.i1.next, %q6b.red.step ]
%q6b.red.i2 = phi i32 [ %q6b.i2, %q6b.role.done ], [ %q6b.red.i2.next, %q6b.red.step ]
%q6b.red.i3 = phi i32 [ %q6b.i3, %q6b.role.done ], [ %q6b.red.i3.next, %q6b.red.step ]
%q6b.red.i4 = phi i32 [ %q6b.i4, %q6b.role.done ], [ %q6b.red.i4.next, %q6b.red.step ]
%q6b.red.i5 = phi i32 [ %q6b.i5, %q6b.role.done ], [ %q6b.red.i5.next, %q6b.red.step ]
%q6b.red.i6 = phi i32 [ %q6b.i6, %q6b.role.done ], [ %q6b.red.i6.next, %q6b.red.step ]
%q6b.red.i7 = phi i32 [ %q6b.i7, %q6b.role.done ], [ %q6b.red.i7.next, %q6b.red.step ]
%q6b.red.more = icmp ugt i32 %q6b.red.offset, 0
br i1 %q6b.red.more, label %q6b.red.step, label %q6b.red.done
q6b.red.step:
%q6b.red.partner.lane = xor i32 %lane, %q6b.red.offset
%q6b.red.partner.index = mul i32 %q6b.red.partner.lane, 4
%q6b.red.i0.bits = bitcast i32 %q6b.red.i0 to float
%q6b.red.i0.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i0.bits, i32 %q6b.red.partner.index)
%q6b.red.i0.partner = bitcast float %q6b.red.i0.partner.bits to i32
%q6b.red.i0.next = add i32 %q6b.red.i0, %q6b.red.i0.partner
%q6b.red.i1.bits = bitcast i32 %q6b.red.i1 to float
%q6b.red.i1.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i1.bits, i32 %q6b.red.partner.index)
%q6b.red.i1.partner = bitcast float %q6b.red.i1.partner.bits to i32
%q6b.red.i1.next = add i32 %q6b.red.i1, %q6b.red.i1.partner
%q6b.red.i2.bits = bitcast i32 %q6b.red.i2 to float
%q6b.red.i2.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i2.bits, i32 %q6b.red.partner.index)
%q6b.red.i2.partner = bitcast float %q6b.red.i2.partner.bits to i32
%q6b.red.i2.next = add i32 %q6b.red.i2, %q6b.red.i2.partner
%q6b.red.i3.bits = bitcast i32 %q6b.red.i3 to float
%q6b.red.i3.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i3.bits, i32 %q6b.red.partner.index)
%q6b.red.i3.partner = bitcast float %q6b.red.i3.partner.bits to i32
%q6b.red.i3.next = add i32 %q6b.red.i3, %q6b.red.i3.partner
%q6b.red.i4.bits = bitcast i32 %q6b.red.i4 to float
%q6b.red.i4.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i4.bits, i32 %q6b.red.partner.index)
%q6b.red.i4.partner = bitcast float %q6b.red.i4.partner.bits to i32
%q6b.red.i4.next = add i32 %q6b.red.i4, %q6b.red.i4.partner
%q6b.red.i5.bits = bitcast i32 %q6b.red.i5 to float
%q6b.red.i5.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i5.bits, i32 %q6b.red.partner.index)
%q6b.red.i5.partner = bitcast float %q6b.red.i5.partner.bits to i32
%q6b.red.i5.next = add i32 %q6b.red.i5, %q6b.red.i5.partner
%q6b.red.i6.bits = bitcast i32 %q6b.red.i6 to float
%q6b.red.i6.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i6.bits, i32 %q6b.red.partner.index)
%q6b.red.i6.partner = bitcast float %q6b.red.i6.partner.bits to i32
%q6b.red.i6.next = add i32 %q6b.red.i6, %q6b.red.i6.partner
%q6b.red.i7.bits = bitcast i32 %q6b.red.i7 to float
%q6b.red.i7.partner.bits = call float @recipe.wave.partner.f32(float %q6b.red.i7.bits, i32 %q6b.red.partner.index)
%q6b.red.i7.partner = bitcast float %q6b.red.i7.partner.bits to i32
%q6b.red.i7.next = add i32 %q6b.red.i7, %q6b.red.i7.partner
%q6b.red.offset.next = lshr i32 %q6b.red.offset, 1
br label %q6b.red.loop
q6b.red.done:
%q6b.d.ptr = getelementptr i8, ptr addrspace(1) %weights, i64 %q6b.byte
%q6b.d.offset = getelementptr i8, ptr addrspace(1) %q6b.d.ptr, i64 208
%q6b.d.bits = load half, ptr addrspace(1) %q6b.d.offset, align 2
%q6b.d = call float @recipe.state.from.f16(half %q6b.d.bits)
%q6b.d8 = load float, ptr addrspace(3) %q6b.q8, align 4
%q6b.scale = call float @recipe.state.mul(float %q6b.d8, float %q6b.d)
%q6b.i0.value = call float @recipe.state.from.s32(i32 %q6b.red.i0)
%q6b.f0.next = call float @recipe.state.madd(float %q6b.f0, float %q6b.scale, float %q6b.i0.value)
%q6b.i1.value = call float @recipe.state.from.s32(i32 %q6b.red.i1)
%q6b.f1.next = call float @recipe.state.madd(float %q6b.f1, float %q6b.scale, float %q6b.i1.value)
%q6b.i2.value = call float @recipe.state.from.s32(i32 %q6b.red.i2)
%q6b.f2.next = call float @recipe.state.madd(float %q6b.f2, float %q6b.scale, float %q6b.i2.value)
%q6b.i3.value = call float @recipe.state.from.s32(i32 %q6b.red.i3)
%q6b.f3.next = call float @recipe.state.madd(float %q6b.f3, float %q6b.scale, float %q6b.i3.value)
%q6b.i4.value = call float @recipe.state.from.s32(i32 %q6b.red.i4)
%q6b.f4.next = call float @recipe.state.madd(float %q6b.f4, float %q6b.scale, float %q6b.i4.value)
%q6b.i5.value = call float @recipe.state.from.s32(i32 %q6b.red.i5)
%q6b.f5.next = call float @recipe.state.madd(float %q6b.f5, float %q6b.scale, float %q6b.i5.value)
%q6b.i6.value = call float @recipe.state.from.s32(i32 %q6b.red.i6)
%q6b.f6.next = call float @recipe.state.madd(float %q6b.f6, float %q6b.scale, float %q6b.i6.value)
%q6b.i7.value = call float @recipe.state.from.s32(i32 %q6b.red.i7)
%q6b.f7.next = call float @recipe.state.madd(float %q6b.f7, float %q6b.scale, float %q6b.i7.value)
br label %q6b.done
q6b.done:
%q6b.block.next = add i32 %q6b.block, 1
br label %q6b.loop
q6b.exit:
%q6b.t0 = call float @recipe.state.add(float %q6b.f0, float %q6b.f4)
%q6b.t1 = call float @recipe.state.add(float %q6b.f1, float %q6b.f5)
%q6b.t2 = call float @recipe.state.add(float %q6b.f2, float %q6b.f6)
%q6b.t3 = call float @recipe.state.add(float %q6b.f3, float %q6b.f7)
%q6b.u0 = call float @recipe.state.add(float %q6b.t0, float %q6b.t2)
%q6b.u1 = call float @recipe.state.add(float %q6b.t1, float %q6b.t3)
%q6b.total = call float @recipe.state.add(float %q6b.u0, float %q6b.u1)
%q6b.owner = icmp eq i32 %lane, 0
%q6b.owned = select i1 %q6b.owner, float %q6b.total, float %state.zero
%q6b.result = select i1 %channel.active, float %q6b.owned, float %state.zero
br label %sum.done
q6.sum.loop:
%q6.slice = phi i32 [ %lane, %q6.check.slices ], [ %q6.slice.next, %q6.slice.ready ]
%q6.sum = phi float [ %state.zero, %q6.check.slices ], [ %q6.sum.next, %q6.slice.ready ]
%q6.slices = udiv i32 %terms, 16
%q6.slice.more = icmp ult i32 %q6.slice, %q6.slices
br i1 %q6.slice.more, label %q6.sum.step, label %q6.sum.done
q6.sum.step:
%q6.block = udiv i32 %q6.slice, 16
%q6.slice.local = urem i32 %q6.slice, 16
%q6.block.wide = zext i32 %q6.block to i64
%q6.block.bytes = mul i64 %q6.block.wide, 210
%q6.byte.offset = add i64 %row.base, %q6.block.bytes
%q6.q8.offset = mul i64 %q6.block.wide, %q8.block.record
%q6.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.shared, i64 %q6.q8.offset
%q6.loaded = call float @recipe.q6k.slice(ptr addrspace(1) %weights, i64 %q6.byte.offset, ptr addrspace(3) %q6.q8.ptr, i32 %q6.slice.local)
%q6.value.active = select i1 %channel.active, float %q6.loaded, float %state.zero
%q6.sum.next = call float @recipe.state.add(float %q6.sum, float %q6.value.active)
%q6.slice.next = add i32 %q6.slice, %width
br label %q6.slice.ready
q6.slice.ready:
br label %q6.sum.loop
q6.sum.done:
br label %sum.done
b32.check:
br i1 %b32.available, label %b32.sum.loop, label %sum.loop
b32.sum.loop:
%b32.slice = phi i32 [ %lane, %b32.check ], [ %b32.slice.next, %b32.slice.ready ]
%b32.sum = phi float [ %state.zero, %b32.check ], [ %b32.sum.next, %b32.slice.ready ]
%b32.slice.width = select i1 %int16, i32 32, i32 16
%b32.slices = udiv i32 %terms, %b32.slice.width
%b32.slice.more = icmp ult i32 %b32.slice, %b32.slices
br i1 %b32.slice.more, label %b32.sum.step, label %b32.sum.done
b32.sum.step:
%b32.row.blocks = udiv i32 %terms, 32
%b32.row.blocks.wide = zext i32 %b32.row.blocks to i64
%b32.channel.row = mul i64 %channel.wide, %b32.row.blocks.wide
%b32.block.half = udiv i32 %b32.slice, 2
%b32.block = select i1 %int16, i32 %b32.slice, i32 %b32.block.half
%b32.slice.local = urem i32 %b32.slice, 2
%b32.block.wide = zext i32 %b32.block to i64
%b32.block.index = add i64 %b32.channel.row, %b32.block.wide
%b32.byte.offset = mul i64 %b32.block.index, %b32.stride
%b32.q8.offset = mul i64 %b32.block.wide, %q8.record
%b32.q8.ptr = getelementptr i8, ptr addrspace(3) %q8.shared, i64 %b32.q8.offset
br i1 %int16, label %b32.load16, label %b32.load8
b32.load16:
%b32.value16 = call float @recipe.block32.i16(i32 %b32.kind, ptr addrspace(1) %weights, i64 %b32.byte.offset, ptr addrspace(3) %b32.q8.ptr, i32 %b32.slice.local)
br label %b32.loaded.done
b32.load8:
%b32.value8 = call float @recipe.block32.slice(i32 %b32.kind, ptr addrspace(1) %weights, i64 %b32.byte.offset, ptr addrspace(3) %b32.q8.ptr, i32 %b32.slice.local)
br label %b32.loaded.done
b32.loaded.done:
%b32.loaded = phi float [ %b32.value16, %b32.load16 ], [ %b32.value8, %b32.load8 ]
%b32.value.active = select i1 %channel.active, float %b32.loaded, float %state.zero
%b32.sum.next = call float @recipe.state.add(float %b32.sum, float %b32.value.active)
%b32.slice.next = add i32 %b32.slice, %width
br label %b32.slice.ready
b32.slice.ready:
br label %b32.sum.loop
b32.sum.done:
br label %sum.done
sum.loop:
%k = phi i32 [ %dense.first, %b32.check ], [ %dense.first, %exact.b32.check ], [ %k.next, %weight.ready ]
%sum = phi float [ %state.zero, %b32.check ], [ %state.zero, %exact.b32.check ], [ %sum.next, %weight.ready ]
%k.more = icmp ult i32 %k, %terms
br i1 %k.more, label %sum.step, label %sum.done
sum.step:
%k.wide = zext i32 %k to i64
%weight.local.index = add i64 %channel.offset, %k.wide
%weight.decode.index = add i64 %weight.base.wide, %weight.local.index
%weight.packed = icmp ne i32 %decode, 0
br i1 %weight.packed, label %weight.packed.load, label %weight.dense.load
weight.dense.load:
%weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %weight.local.index
%weight.dense.model = load i8, ptr addrspace(1) %weight.ptr, align 1
br label %weight.ready
weight.packed.load:
%weight.packed.model = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %weight.decode.index, i32 %decode)
br label %weight.ready
weight.ready:
%weight.model = phi i8 [ %weight.dense.model, %weight.dense.load ], [ %weight.packed.model, %weight.packed.load ]
%input.channel = zext i32 %k to i64
%input.offset = mul i64 %input.channel, %in.length.wide
%input.index = add i64 %input.offset, %position
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %input.index
%input.model = load i8, ptr addrspace(1) %input.ptr, align 1
%weight.wide = call float @recipe.decode(i8 %weight.model)
%input.wide = call float @recipe.decode(i8 %input.model)
%product.raw = call float @recipe.state.mul(float %weight.wide, float %input.wide)
%product = select i1 %channel.active, float %product.raw, float %state.zero
%sum.next = call float @recipe.state.add(float %sum, float %product)
%k.next = add i32 %k, %dense.width
br label %sum.loop
sum.done:
%sum.final = phi float [ %sum, %sum.loop ], [ %q4b.result, %q4b.exit ], [ %q4.sum, %q4.sum.done ], [ %q6.sum, %q6.sum.done ], [ %q6b.result, %q6b.exit ], [ %b32.sum, %b32.sum.done ], [ %exact.q4.sum, %exact.q4.done ], [ %exact.q6.sum, %exact.q6.done ], [ %exact.b32.sum, %exact.b32.done ]
%sum.partition = phi i32 [ 0, %sum.loop ], [ 0, %q4b.exit ], [ 0, %q4.sum.done ], [ 0, %q6.sum.done ], [ 0, %q6b.exit ], [ 0, %b32.sum.done ], [ %exact.lane, %exact.q4.done ], [ %exact.lane, %exact.q6.done ], [ %exact.lane, %exact.b32.done ]
%exact.cpu = and i1 %stage.available, %exact.serial
br i1 %exact.cpu, label %exact.serial.store, label %reduce.begin
exact.serial.store:
%exact.partial.ptr = getelementptr [32 x float], ptr %exact.partials, i32 0, i32 %sum.partition
store float %sum.final, ptr %exact.partial.ptr, align 4
%exact.lane.next = add i32 %sum.partition, 1
%exact.lane.more = icmp ult i32 %exact.lane.next, 32
br i1 %exact.lane.more, label %exact.partition, label %exact.tree
exact.tree:
%exact.offset = phi i32 [ 16, %exact.serial.store ], [ %exact.offset.next, %exact.tree.done ]
br label %exact.tree.loop
exact.tree.loop:
%exact.index = phi i32 [ 0, %exact.tree ], [ %exact.index.next, %exact.tree.step ]
%exact.more = icmp ult i32 %exact.index, %exact.offset
br i1 %exact.more, label %exact.tree.step, label %exact.tree.done
exact.tree.step:
%exact.partner = add i32 %exact.index, %exact.offset
%exact.left.ptr = getelementptr [32 x float], ptr %exact.partials, i32 0, i32 %exact.index
%exact.right.ptr = getelementptr [32 x float], ptr %exact.partials, i32 0, i32 %exact.partner
%exact.left = load float, ptr %exact.left.ptr, align 4
%exact.right = load float, ptr %exact.right.ptr, align 4
%exact.added = call float @recipe.state.add(float %exact.left, float %exact.right)
store float %exact.added, ptr %exact.left.ptr, align 4
%exact.index.next = add i32 %exact.index, 1
br label %exact.tree.loop
exact.tree.done:
%exact.offset.next = udiv i32 %exact.offset, 2
%exact.tree.more = icmp ugt i32 %exact.offset.next, 0
br i1 %exact.tree.more, label %exact.tree, label %exact.result
exact.result:
%exact.total = load float, ptr %exact.partials, align 4
br label %reduce.done
reduce.begin:
%reduce.physical = udiv i32 %width, 2
%reduce.offset.initial = select i1 %stage.available, i32 16, i32 %reduce.physical
br label %reduce.loop
reduce.loop:
%reduce.offset = phi i32 [ %reduce.offset.initial, %reduce.begin ], [ %reduce.offset.next, %reduce.step ]
%reduced = phi float [ %sum.final, %reduce.begin ], [ %reduced.next, %reduce.step ]
%reduce.more = icmp ugt i32 %reduce.offset, 0
br i1 %reduce.more, label %reduce.step, label %reduce.done
reduce.step:
%partner.lane = xor i32 %lane, %reduce.offset
%partner.index = mul i32 %partner.lane, 4
%partner = call float @recipe.wave.partner(float %reduced, i32 %partner.index)
%reduced.next = call float @recipe.state.add(float %reduced, float %partner)
%reduce.offset.next = udiv i32 %reduce.offset, 2
br label %reduce.loop
reduce.done:
%sum.reduced = phi float [ %reduced, %reduce.loop ], [ %exact.total, %exact.result ]
%owner = icmp eq i32 %lane, 0
%store = and i1 %owner, %channel.active
br i1 %store, label %bias.select, label %job.done
bias.select:
%bias.base = mul i32 %out.channels, %terms
%bias.index = add i32 %bias.base, %channel
%bias.wide.index = zext i32 %bias.index to i64
br i1 %has.bias, label %bias.load, label %bias.zero
bias.load:
%bias.decode.index = add i64 %weight.base.wide, %bias.wide.index
%bias.packed = icmp ne i32 %decode, 0
br i1 %bias.packed, label %bias.packed.load, label %bias.dense.load
bias.dense.load:
%bias.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %bias.wide.index
%bias.dense.model = load i8, ptr addrspace(1) %bias.ptr, align 1
br label %bias.ready
bias.packed.load:
%bias.packed.model = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %bias.decode.index, i32 %decode)
br label %bias.ready
bias.zero:
%bias.zero.model = call i8 @recipe.encode(float %state.zero)
br label %bias.ready
bias.ready:
%bias.model = phi i8 [ %bias.dense.model, %bias.dense.load ], [ %bias.packed.model, %bias.packed.load ], [ %bias.zero.model, %bias.zero ]
%bias.wide = call float @recipe.decode(i8 %bias.model)
%sum.bias = call float @recipe.state.add(float %sum.reduced, float %bias.wide)
%bias.now = and i1 %has.bias, %chunk.first
%sum.value = select i1 %bias.now, float %sum.bias, float %sum.reduced
%output.channel = zext i32 %channel to i64
%out.length.wide = zext i32 %out.length to i64
%output.channel.base = mul i64 %output.channel, %out.length.wide
%output.index = add i64 %output.channel.base, %position
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.index
; A later chunk adds the partial the earlier chunks left in the output; the
; first chunk reads nothing there.
br i1 %chunk.first, label %prior.none, label %prior.load
prior.load:
%prior.model = load i8, ptr addrspace(1) %output.ptr, align 1
%prior.wide = call float @recipe.decode(i8 %prior.model)
%sum.carried = call float @recipe.state.add(float %sum.value, float %prior.wide)
br label %prior.none
prior.none:
%sum.total = phi float [ %sum.value, %bias.ready ], [ %sum.carried, %prior.load ]
%result.model = call i8 @recipe.encode(float %sum.total)
%result.positive = call i1 @recipe.ogt(i8 %result.model, i8 0)
%result.activated = select i1 %result.positive, i8 %result.model, i8 0
%relu.now = and i1 %relu, %chunk.last
%result = select i1 %relu.now, i8 %result.activated, i8 %result.model
store i8 %result, ptr addrspace(1) %output.ptr, align 1
br label %job.done
job.done:
%job.next = add i32 %job, %groups
br label %job.loop
chunk.done:
call void @llvm.nvvm.barrier0()
%chunk.next = add i32 %chunk.base, %chunk.terms
%chunk.more = icmp ult i32 %chunk.next, %terms
br i1 %chunk.more, label %chunk.loop, label %position.done
position.done:
%position.next = add i32 %position.index, 1
br label %position.loop
exit:
ret void
}
define internal void @contraction_forward_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel,
i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode ) RECIPE_CONTRACTION_BODY { entry:
%one = icmp eq i32 %out.span, 1
%kernel.zero = icmp eq i32 %kernel, 0
%rows.one = icmp eq i32 %rows, 1
%reverse.off = xor i1 %reverse, true
%accumulate.off = xor i1 %accumulate, true
%relu.off = xor i1 %relu, true
%transpose.off = xor i1 %transpose, true
%fast.a = and i1 %one, %kernel.zero
%fast.b = and i1 %fast.a, %rows.one
%fast.c = and i1 %fast.b, %reverse.off
%fast.d = and i1 %fast.c, %accumulate.off
%fast.e = and i1 %fast.d, %relu.off
%fast.f = and i1 %fast.e, %transpose.off
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%width = call i32 @recipe.wavefront.width()
%waves = udiv i32 %block, %width
%waves.ok = icmp ugt i32 %waves, 0
%width.ok = icmp ugt i32 %width, 1
%wave.available = and i1 %waves.ok, %width.ok
; An int sum is an int dot at every position, not only the single-position
; decode: block weights whose dot the wave body has send the whole span there.
%blocked.q4 = call i1 @recipe.model.q4k(i32 %decode)
%blocked.q6 = call i1 @recipe.model.q6k(i32 %decode)
%blocked.kind = call i32 @recipe.model.block32(i32 %decode)
%blocked.b32 = icmp ne i32 %blocked.kind, 0
%blocked.k = or i1 %blocked.q4, %blocked.q6
%blocked.rem256 = urem i32 %in.channels, 256
%blocked.a256 = icmp eq i32 %blocked.rem256, 0
%blocked.rem32 = urem i32 %in.channels, 32
%blocked.a32 = icmp eq i32 %blocked.rem32, 0
%blocked.ok.k = and i1 %blocked.k, %blocked.a256
%blocked.ok.b32 = and i1 %blocked.b32, %blocked.a32
%blocked = or i1 %blocked.ok.k, %blocked.ok.b32
%span.ok = or i1 %one, %blocked
%wave.a = and i1 %span.ok, %kernel.zero
%wave.b = and i1 %wave.a, %rows.one
%wave.c = and i1 %wave.b, %reverse.off
%wave.d = and i1 %wave.c, %accumulate.off
%wave.e = and i1 %wave.d, %relu.off
%wave.f = and i1 %wave.e, %transpose.off
%wave.or.blocked = or i1 %wave.available, %blocked
%wave.fast = and i1 %wave.f, %wave.or.blocked
br i1 %wave.fast, label %wave, label %scalar.check
wave:
call void @contraction_forward_gemv_wave_body(ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel, i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode)
ret void
scalar.check:
br i1 %fast.f, label %scalar, label %gemm
scalar:
call void @contraction_forward_gemv_body(ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel, i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode)
ret void
gemm:
call void @contraction_forward_gemm_body(ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel, i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode)
ret void
}
define internal void @contraction_forward_gemm_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel,
i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode ) #1 { entry:
; A packed node passes a nonzero decoder selector and keeps its weights in the stored
; representation, so every weight read decodes one element instead of loading one.
%weight.packed = icmp ne i32 %decode, 0 %weight.dense = xor i1 %weight.packed, true
; The running sums live in the arithmetic type for the whole K extent and are
; rounded to the model type once, at the store. Staging the operands in tiles
; therefore cannot move a rounding point.
%sums = alloca [RECIPE_REGISTER_COUNT x float], align 4 %state.zero = call float @recipe.state.from.u1(i1 false) %lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() %group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() %block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %groups = udiv i32 %threads, %block
%in.channels.wide = zext i32 %in.channels to i64 %in.length.wide = zext i32 %in.length to i64 %out.channels.wide = zext i32 %out.channels to i64 %out.length.wide = zext i32 %out.length to i64 %rows.wide = zext i32 %rows to i64 %out.begin.wide = zext i32 %out.begin to i64 %out.span.wide = zext i32 %out.span to i64 %weight.base.wide = add i64 %weight.base, 0
%in.elements = mul i32 %in.channels, %in.length %in.elements.wide = mul i64 %in.channels.wide, %in.length.wide %out.elements.wide = mul i64 %out.channels.wide, %out.length.wide %is.conv = icmp ne i32 %kernel, 0 %span = select i1 %is.conv, i32 %kernel, i32 1 %terms = mul i32 %in.channels, %span %terms.wide = zext i32 %terms to i64 %m.total = mul i32 %rows, %out.span
%m.short = icmp ult i32 %tile.m, %m.total %m.tile.clamped = select i1 %m.short, i32 %tile.m, i32 %m.total %m.tile.empty = icmp eq i32 %m.tile.clamped, 0 %m.tile = select i1 %m.tile.empty, i32 1, i32 %m.tile.clamped %n.short = icmp ult i32 %tile.n, %out.channels %n.tile = select i1 %n.short, i32 %tile.n, i32 %out.channels %k.short = icmp ult i32 %tile.k, %terms %k.tile = select i1 %k.short, i32 %tile.k, i32 %terms
%m.adjusted = add i32 %m.total, %m.tile %m.numerator = sub i32 %m.adjusted, 1 %m.tiles = udiv i32 %m.numerator, %m.tile %n.adjusted = add i32 %out.channels, %n.tile %n.numerator = sub i32 %n.adjusted, 1 %n.tiles = udiv i32 %n.numerator, %n.tile %jobs = mul i32 %m.tiles, %n.tiles br label %job.loop job.loop:
%job = phi i32 [ %group, %entry ], [ %job.next, %job.done ] %job.more = icmp ult i32 %job, %jobs br i1 %job.more, label %job.step, label %exit job.step:
%m.group.short = icmp ult i32 %m.tiles, 4 %m.group.limit = select i1 %m.group.short, i32 %m.tiles, i32 4 %group.width = mul i32 %m.group.limit, %n.tiles %group.index = udiv i32 %job, %group.width %m.group.base = mul i32 %group.index, %m.group.limit %m.group.remaining = sub i32 %m.tiles, %m.group.base %m.group.tail = icmp ult i32 %m.group.remaining, %m.group.limit %m.group.count = select i1 %m.group.tail, i32 %m.group.remaining, i32 %m.group.limit %group.local = urem i32 %job, %group.width %m.group.local = urem i32 %group.local, %m.group.count %m.tile.index = add i32 %m.group.base, %m.group.local %n.tile.index = udiv i32 %group.local, %m.group.count %m.base = mul i32 %m.tile.index, %m.tile %n.base = mul i32 %n.tile.index, %n.tile
%m.remaining = sub i32 %m.total, %m.base %m.partial = icmp ult i32 %m.remaining, %m.tile %m.count = select i1 %m.partial, i32 %m.remaining, i32 %m.tile %n.remaining = sub i32 %out.channels, %n.base %n.partial = icmp ult i32 %n.remaining, %n.tile %n.count = select i1 %n.partial, i32 %n.remaining, i32 %n.tile %m.base.wide = zext i32 %m.base to i64 %n.base.wide = zext i32 %n.base to i64
%m.lanes.adjusted = add i32 %m.count, RECIPE_REGISTER_M %m.lanes.numerator = sub i32 %m.lanes.adjusted, 1 %m.lanes = udiv i32 %m.lanes.numerator, RECIPE_REGISTER_M %n.lanes.adjusted = add i32 %n.count, RECIPE_REGISTER_N %n.lanes.numerator = sub i32 %n.lanes.adjusted, 1 %n.lanes = udiv i32 %n.lanes.numerator, RECIPE_REGISTER_N
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
%output.m.base = mul i32 %lane.m, RECIPE_REGISTER_M %output.n.base = mul i32 %lane.n, RECIPE_REGISTER_N br label %sum.init.loop sum.init.loop:
%sum.init = phi i32 [ 0, %job.step ], [ %sum.init.next, %sum.init.step ] %sum.init.more = icmp ult i32 %sum.init, RECIPE_REGISTER_COUNT br i1 %sum.init.more, label %sum.init.step, label %sum.init.done
sum.init.step: %sum.init.ptr = getelementptr [RECIPE_REGISTER_COUNT x float], ptr %sums, i32 0, i32 %sum.init store float %state.zero, ptr %sum.init.ptr, align 4 %sum.init.next = add i32 %sum.init, 1 br label %sum.init.loop
sum.init.done:
br label %tile.loop
tile.loop:
%term.base = phi i32 [ 0, %sum.init.done ], [ %term.next, %tile.done ] %term.base.wide = zext i32 %term.base to i64 %k.remaining = sub i32 %terms, %term.base %k.partial = icmp ult i32 %k.remaining, %k.tile %k.count = select i1 %k.partial, i32 %k.remaining, i32 %k.tile
%a.project = icmp eq i32 %span, 1
%a.unit = icmp eq i32 %in.length, 1
%a.contiguous = and i1 %a.project, %a.unit
%a.fragment.remainder = urem i32 %k.count, RECIPE_FRAGMENT_K
%a.fragment.full = icmp eq i32 %a.fragment.remainder, 0
%a.gate = and i1 %reverse, %relu %a.ungated = xor i1 %a.gate, true %a.vector.shape = and i1 %a.contiguous, %a.fragment.full %a.vector = and i1 %a.vector.shape, %a.ungated
%a.width = select i1 %a.vector, i32 RECIPE_FRAGMENT_K, i32 1
%a.columns = udiv i32 %k.count, %a.width
%b.fragment.remainder = urem i32 %k.count, RECIPE_FRAGMENT_K
%b.fragment.full = icmp eq i32 %b.fragment.remainder, 0 %b.direct = xor i1 %transpose, true %b.contiguous = and i1 %b.fragment.full, %b.direct %b.vector = and i1 %b.contiguous, %weight.dense
%b.width = select i1 %b.vector, i32 RECIPE_FRAGMENT_K, i32 1
%b.rows = udiv i32 %k.count, %b.width
%a.count = mul i32 %m.count, %a.columns %b.count = mul i32 %n.count, %b.rows %load.count = add i32 %a.count, %b.count br label %load.loop load.loop:
%load = phi i32 [ %lid, %tile.loop ], [ %load.next, %load.advance ] %load.more = icmp ult i32 %load, %load.count br i1 %load.more, label %load.classify, label %load.done load.classify: %load.a = icmp ult i32 %load, %a.count br i1 %load.a, label %load.a.step, label %load.b.step
load.a.step: %a.m = udiv i32 %load, %a.columns %a.column = urem i32 %load, %a.columns %a.k = mul i32 %a.column, %a.width %a.global = add i32 %m.base, %a.m %a.row = udiv i32 %a.global, %out.span %a.position.local = urem i32 %a.global, %out.span %a.position = add i32 %a.position.local, %out.begin %a.row.wide = zext i32 %a.row to i64 %a.position.wide = zext i32 %a.position to i64 %a.row.base = mul i64 %a.row.wide, %in.elements.wide %a.term = add i32 %term.base, %a.k %a.term.wide = zext i32 %a.term to i64
%a.tile.index = call i32 @contraction_vector_a_index(i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br i1 %a.vector, label %load.a.vector, label %load.a.scalar
load.a.vector:
%a.vector.index = add i64 %a.row.base, %a.term.wide
%a.vector.source = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %a.vector.index
%a.vector.value = load <RECIPE_FRAGMENT_K x i8>, ptr addrspace(1) %a.vector.source, align 1
call void @contraction_stage_a_fragment(<RECIPE_FRAGMENT_K x i8> %a.vector.value, i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br label %load.advance
load.a.scalar:
%a.loaded = call i8 @contraction_input( ptr addrspace(1) %input, i64 %a.row.base, i32 %a.position, i32 %a.term, i32 %span, i32 %in.length, i1 %is.conv )
br i1 %a.gate, label %load.a.activation, label %load.a.ready
load.a.activation:
%a.activation.channel = mul i64 %a.term.wide, %in.length.wide %a.activation.local = add i64 %a.activation.channel, %a.position.wide %a.activation.index = add i64 %a.row.base, %a.activation.local %a.activation.ptr = getelementptr inbounds i8, ptr addrspace(1) %activation, i64 %a.activation.index %a.activation.value = load i8, ptr addrspace(1) %a.activation.ptr, align 2 %a.activation.positive = call i1 @recipe.ogt(i8 %a.activation.value, i8 0) %a.gated = select i1 %a.activation.positive, i8 %a.loaded, i8 0
br label %load.a.ready
load.a.ready:
%a.value = phi i8 [ %a.loaded, %load.a.scalar ], [ %a.gated, %load.a.activation ]
br label %load.store
load.b.step: %b.local = sub i32 %load, %a.count %b.n = udiv i32 %b.local, %b.rows %b.row = urem i32 %b.local, %b.rows %b.k = mul i32 %b.row, %b.width %b.n.wide = zext i32 %b.n to i64 %b.k.wide = zext i32 %b.k to i64 %b.channel = add i64 %n.base.wide, %b.n.wide %b.channel.base = mul i64 %b.channel, %terms.wide %b.term = add i64 %term.base.wide, %b.k.wide
%b.direct.index = add i64 %b.channel.base, %b.term %b.out.channels.wide = zext i32 %out.channels to i64 %b.transpose.base = mul i64 %b.term, %b.out.channels.wide %b.transpose.index = add i64 %b.transpose.base, %b.channel %b.index = select i1 %transpose, i64 %b.transpose.index, i64 %b.direct.index %b.tile.base = mul i32 %tile.m, %tile.k %b.tile.local = call i32 @contraction_vector_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k) %b.tile.index = add i32 %b.tile.base, %b.tile.local
br i1 %b.vector, label %load.b.vector, label %load.b.scalar
load.b.vector:
%b.vector.source = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %b.index
%b.vector.value = load <RECIPE_FRAGMENT_K x i8>, ptr addrspace(1) %b.vector.source, align 1
call void @contraction_stage_b_terms(<RECIPE_FRAGMENT_K x i8> %b.vector.value, i32 %b.k, i32 %b.n, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %load.advance
load.b.scalar:
br i1 %weight.packed, label %load.b.packed, label %load.b.direct
load.b.direct:
%b.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %b.index
%b.loaded = load i8, ptr addrspace(1) %b.ptr, align 1
br label %load.b.ready
load.b.packed:
%b.decode.index = add i64 %weight.base.wide, %b.index
%b.decoded = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %b.decode.index, i32 %decode)
br label %load.b.ready
load.b.ready:
%b.value = phi i8 [ %b.loaded, %load.b.direct ], [ %b.decoded, %load.b.packed ]
br label %load.store
load.store: %load.value = phi i8 [ %a.value, %load.a.ready ], [ %b.value, %load.b.ready ] %load.tile.index = phi i32 [ %a.tile.index, %load.a.ready ], [ %b.tile.index, %load.b.ready ] %load.tile.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %load.tile.index store i8 %load.value, ptr addrspace(3) %load.tile.ptr, align 1
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
%store.register = phi i32 [ 0, %accumulate.done ], [ %store.register.next, %store.next ] %store.more = icmp ult i32 %store.register, RECIPE_REGISTER_COUNT br i1 %store.more, label %store.check, label %job.done
store.check: %store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %store.register, i32 %m.lanes) %store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %store.register, i32 %m.lanes) %store.register.valid = call i1 @contraction_output_register_valid(i32 %store.register)
%store.output.m.valid = icmp ult i32 %store.output.m.raw, %m.count %store.output.n.valid = icmp ult i32 %store.output.n.raw, %n.count %store.output.valid = and i1 %store.output.m.valid, %store.output.n.valid %store.lane.active = and i1 %method.store, %store.output.valid %store.active = and i1 %store.lane.active, %store.register.valid br i1 %store.active, label %store, label %store.next
store: %store.channel = add i32 %n.base, %store.output.n.raw %store.m.global = add i32 %m.base, %store.output.m.raw %store.m.global.wide = zext i32 %store.m.global to i64 %store.span.wide = zext i32 %out.span to i64 %store.row.wide = udiv i64 %store.m.global.wide, %store.span.wide %store.position.local.wide = urem i64 %store.m.global.wide, %store.span.wide %store.begin.wide = zext i32 %out.begin to i64 %store.position.wide = add i64 %store.position.local.wide, %store.begin.wide %store.output.row.base = mul i64 %store.row.wide, %out.elements.wide
%store.channel.wide = zext i32 %store.channel to i64 %store.output.channel.base = mul i64 %store.channel.wide, %out.length.wide %store.output.local = add i64 %store.output.channel.base, %store.position.wide %store.output.index = add i64 %store.output.row.base, %store.output.local %store.output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %store.output.index
%store.bias.base = mul i64 %out.channels.wide, %terms.wide %store.channel.offset = zext i32 %store.channel to i64 %store.bias.index = add i64 %store.bias.base, %store.channel.offset
br i1 %weight.packed, label %store.bias.packed, label %store.bias.direct
store.bias.direct:
%store.bias.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %store.bias.index
%store.bias.loaded = load i8, ptr addrspace(1) %store.bias.ptr, align 1
br label %store.bias.ready
store.bias.packed:
%store.bias.decode.index = add i64 %weight.base.wide, %store.bias.index
%store.bias.decoded = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %store.bias.decode.index, i32 %decode)
br label %store.bias.ready
store.bias.ready:
%store.bias = phi i8 [ %store.bias.loaded, %store.bias.direct ], [ %store.bias.decoded, %store.bias.packed ]
%store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x float], ptr %sums, i32 0, i32 %store.register %store.sum.wide = load float, ptr %store.sum.ptr, align 4 %store.sum = call i8 @recipe.encode(float %store.sum.wide)
%store.biased = call i8 @recipe.add(i8 %store.sum, i8 %store.bias) %store.raw = select i1 %has.bias, i8 %store.biased, i8 %store.sum %store.forward = xor i1 %reverse, true %store.activate = and i1 %relu, %store.forward %store.positive = call i1 @recipe.ogt(i8 %store.raw, i8 0) %store.activated = select i1 %store.positive, i8 %store.raw, i8 0 %store.result = select i1 %store.activate, i8 %store.activated, i8 %store.raw %store.prior = load i8, ptr addrspace(1) %store.output.ptr, align 2 %store.accumulated = call i8 @recipe.add(i8 %store.prior, i8 %store.result) %store.value = select i1 %accumulate, i8 %store.accumulated, i8 %store.result store i8 %store.value, ptr addrspace(1) %store.output.ptr, align 1 br label %store.next
store.next: %store.register.next = add i32 %store.register, 1 br label %store.loop job.done: %job.next = add i32 %job, %groups br label %job.loop exit: ret void }
define internal void @pool_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %p, i32 %from, i32 %to, i32 %size, i32 %channels, i1 %store.index ) #1 { entry: %from.wide = zext i32 %from to i64 %to.wide = zext i32 %to to i64 %size.wide = zext i32 %size to i64 %channels.wide = zext i32 %channels to i64
%length = udiv i64 %from.wide, %channels.wide %pooled.length = udiv i64 %to.wide, %channels.wide %row = udiv i64 %p, %to.wide %out = urem i64 %p, %to.wide
%channel = udiv i64 %out, %pooled.length %spatial = urem i64 %out, %pooled.length %start = mul i64 %spatial, %size.wide
%candidate.end = add i64 %start, %size.wide %short = icmp ult i64 %candidate.end, %length
%end = select i1 %short, i64 %candidate.end, i64 %length %row.base = mul i64 %row, %from.wide
%channel.local = mul i64 %channel, %length %input.base = add i64 %row.base, %channel.local br label %loop loop:
%i = phi i64 [ %start, %entry ], [ %next, %step ]
%maximum = phi i8 [ 254, %entry ], [ %maximum.next, %step ]
%maximum.index = phi i64 [ %start, %entry ], [ %maximum.index.next, %step ] %more = icmp ult i64 %i, %end
br i1 %more, label %step, label %done step: %index = add i64 %input.base, %i
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %index
%value = load i8, ptr addrspace(1) %input.ptr, align 1 %greater = call i1 @recipe.ogt(i8 %value, i8 %maximum)
%maximum.next = select i1 %greater, i8 %value, i8 %maximum
%maximum.index.next = select i1 %greater, i64 %index, i64 %maximum.index %next = add i64 %i, 1 br label %loop done:
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p
store i8 %maximum, ptr addrspace(1) %output.ptr, align 1
br i1 %store.index, label %store.argmax, label %pool.exit
store.argmax:
%context.ptr = getelementptr inbounds i64, ptr addrspace(1) %context, i64 %p
store i64 %maximum.index, ptr addrspace(1) %context.ptr, align 1
br label %pool.exit
pool.exit:
ret void }
; Rotary embedding over the first %rotated channels of a fused QKV row: inside
; each head the channel pairs (i, i + dims/2) below %dims rotate by
; position * base^(-2i/dims). With %reverse the transpose rotation is added
; into %output, which makes the same body the adjoint pass.
define internal i32 @rope_position(ptr addrspace(1) %context, i1 %sectioned, i64 %p, i64 %per.row, i64 %length.wide, i64 %position, i32 %index.i32, i32 %position.origin) #1 {
entry:
%position.local = trunc i64 %position to i32 %position.i32 = add i32 %position.local, %position.origin
br i1 %sectioned, label %axis.pick, label %plain
plain:
ret i32 %position.i32
axis.pick:
%axes.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 0 %axes = load i32, ptr addrspace(1) %axes.ptr, align 4
%interleave.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 1 %interleave.word = load i32, ptr addrspace(1) %interleave.ptr, align 4 %interleave = icmp ne i32 %interleave.word, 0
%size.0.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 2 %size.0 = load i32, ptr addrspace(1) %size.0.ptr, align 4
%size.1.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 3 %size.1 = load i32, ptr addrspace(1) %size.1.ptr, align 4
%size.2.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 4 %size.2 = load i32, ptr addrspace(1) %size.2.ptr, align 4
%size.3.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 5 %size.3 = load i32, ptr addrspace(1) %size.3.ptr, align 4
%bound.1 = add i32 %size.0, %size.1 %bound.2 = add i32 %bound.1, %size.2 %total = add i32 %bound.2, %size.3
%total.empty = icmp eq i32 %total, 0 %total.safe = select i1 %total.empty, i32 1, i32 %total
%sector = urem i32 %index.i32, %total.safe
%last.axis = sub i32 %axes, 1
%blocked.in.0 = icmp ult i32 %sector, %size.0 %blocked.in.1 = icmp ult i32 %sector, %bound.1 %blocked.in.2 = icmp ult i32 %sector, %bound.2
%blocked.from.2 = select i1 %blocked.in.2, i32 2, i32 3 %blocked.from.1 = select i1 %blocked.in.1, i32 1, i32 %blocked.from.2 %blocked.raw = select i1 %blocked.in.0, i32 0, i32 %blocked.from.1
%blocked.over = icmp ugt i32 %blocked.raw, %last.axis %blocked.axis = select i1 %blocked.over, i32 %last.axis, i32 %blocked.raw
%turn = urem i32 %sector, %axes
%turn.is.0 = icmp eq i32 %turn, 0 %turn.is.1 = icmp eq i32 %turn, 1 %turn.is.2 = icmp eq i32 %turn, 2
%turn.size.high = select i1 %turn.is.2, i32 %size.2, i32 %size.3 %turn.size.low = select i1 %turn.is.0, i32 %size.0, i32 %size.1 %turn.size = select i1 %turn.is.1, i32 %size.1, i32 %turn.size.high
%turn.size.pick = select i1 %turn.is.0, i32 %turn.size.low, i32 %turn.size
%turn.limit = mul i32 %axes, %turn.size.pick %turn.inside = icmp ult i32 %sector, %turn.limit %interleaved.axis = select i1 %turn.inside, i32 %turn, i32 %last.axis
%axis = select i1 %interleave, i32 %interleaved.axis, i32 %blocked.axis
%row = udiv i64 %p, %per.row %axes.wide = zext i32 %axes to i64 %axis.wide = zext i32 %axis to i64
%row.axes = mul i64 %row, %axes.wide %row.axis = add i64 %row.axes, %axis.wide %row.axis.base = mul i64 %row.axis, %length.wide %origin.wide = zext i32 %position.origin to i64 %column = add i64 %position, %origin.wide %table.local = add i64 %row.axis.base, %column
%table.index = add i64 %table.local, 6 %table.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %table.index %axis.position = load i32, ptr addrspace(1) %table.ptr, align 4
ret i32 %axis.position
}
; Canonical scalar pair arithmetic shared by rotary tensors and pooled keys.
; The base argument holds a direct base or theta scale according to chain.on.
define internal float @rope_pair(float %value, float %other, float %base.wide, float %mscale.wide,
float %factor.wide, float %low.wide, float %high.wide, float %frequency.factor,
i64 %index, i32 %dims, i32 %position.i32, i1 %upper, i1 %reverse, i1 %chain.on) #1 {
entry:
%zero = call float @recipe.state.from.u1(i1 false)
%one = call float @recipe.state.from.u1(i1 true)
%two.index = mul i64 %index, 2
%two.index.i32 = trunc i64 %two.index to i32
%index.i32 = trunc i64 %index to i32
%two.index.value = call float @recipe.state.from.u32(i32 %two.index.i32)
%index.value = call float @recipe.state.from.u32(i32 %index.i32)
%dims.value = call float @recipe.state.from.u32(i32 %dims) %ratio = call float @recipe.state.div(float %two.index.value, float %dims.value)
%log.base = call float @recipe.state.log(float %base.wide) %exponent.positive = call float @recipe.state.mul(float %ratio, float %log.base)
%exponent = call float @recipe.state.neg(float %exponent.positive) %frequency.raw = call float @recipe.state.exp(float %exponent)
%frequency.extrap = call float @recipe.state.div(float %frequency.raw, float %frequency.factor)
%yarn.on = call i1 @recipe.state.ogt(float %factor.wide, float %one)
%ramp.span = call float @recipe.state.sub(float %high.wide, float %low.wide)
%ramp.offset = call float @recipe.state.sub(float %index.value, float %low.wide)
%ramp.raw = call float @recipe.state.div(float %ramp.offset, float %ramp.span)
%ramp.low = call i1 @recipe.state.ogt(float %zero, float %ramp.raw)
%ramp.clamped.low = select i1 %ramp.low, float %zero, float %ramp.raw
%ramp.high = call i1 @recipe.state.ogt(float %ramp.clamped.low, float %one)
%ramp = select i1 %ramp.high, float %one, float %ramp.clamped.low
%interpolated = call float @recipe.state.div(float %frequency.extrap, float %factor.wide)
%ramp.inverse = call float @recipe.state.sub(float %one, float %ramp)
%extrapolated.part = call float @recipe.state.mul(float %ramp.inverse, float %frequency.extrap)
%interpolated.part = call float @recipe.state.mul(float %ramp, float %interpolated)
%blended = call float @recipe.state.add(float %extrapolated.part, float %interpolated.part)
%frequency = select i1 %yarn.on, float %blended, float %frequency.extrap
%position.value = call float @recipe.state.from.u32(i32 %position.i32) %angle = call float @recipe.state.mul(float %position.value, float %frequency)
%cos = call float @recipe.state.cos(float %angle) %sin = call float @recipe.state.sin(float %angle) %sin.negative = call float @recipe.state.neg(float %sin)
%sin.signed = select i1 %reverse, float %sin.negative, float %sin %sin.signed.negative = call float @recipe.state.neg(float %sin.signed)
%sin.term = select i1 %upper, float %sin.signed, float %sin.signed.negative
%cos.part = call float @recipe.state.mul(float %value, float %cos) %sin.part = call float @recipe.state.mul(float %other, float %sin.term)
%rotated.raw = call float @recipe.state.add(float %cos.part, float %sin.part)
%rotated.direct = call float @recipe.state.mul(float %rotated.raw, float %mscale.wide)
; The chain angle, llama.cpp's rope: the pair's angle is the position times
; base^(-2/dims) taken one product at a time in the state, the yarn blend
; falls on the angle, the cosine and sine each carry the magnitude scale, and
; the rotation is two fused multiply-adds against the libm trig of the CPU.
; Under the chain the base argument already holds base^(-2/dims), taken on
; the machine with the CPU's powf so every backend chains the same bits.
br i1 %chain.on, label %chain, label %finish.direct
chain:
%chain.scale = call float @recipe.state.add(float %base.wide, float %zero)
br label %chain.loop
chain.loop:
%chain.i = phi i64 [ 0, %chain ], [ %chain.i.next, %chain.step ]
%chain.theta = phi float [ %position.value, %chain ], [ %chain.theta.next, %chain.step ]
%chain.more = icmp ult i64 %chain.i, %index
br i1 %chain.more, label %chain.step, label %chain.done
chain.step:
%chain.theta.next = call float @recipe.state.mul(float %chain.theta, float %chain.scale)
%chain.i.next = add i64 %chain.i, 1
br label %chain.loop
chain.done:
%chain.theta.extrap = call float @recipe.state.div(float %chain.theta, float %frequency.factor)
%chain.freq.scale = call float @recipe.state.div(float %one, float %factor.wide)
%chain.interp = call float @recipe.state.mul(float %chain.freq.scale, float %chain.theta.extrap)
%chain.mix = call float @recipe.state.sub(float %one, float %ramp)
%chain.mix.inverse = call float @recipe.state.sub(float %one, float %chain.mix)
%chain.extrap.part = call float @recipe.state.mul(float %chain.theta.extrap, float %chain.mix)
%chain.angle.yarn = call float @recipe.state.madd(float %chain.extrap.part, float %chain.interp, float %chain.mix.inverse)
%chain.angle = select i1 %yarn.on, float %chain.angle.yarn, float %chain.theta.extrap
%chain.cos.raw = call float @recipe.libm.cos(float %chain.angle)
%chain.sin.raw = call float @recipe.libm.sin(float %chain.angle)
%chain.cos = call float @recipe.state.mul(float %chain.cos.raw, float %mscale.wide)
%chain.sin = call float @recipe.state.mul(float %chain.sin.raw, float %mscale.wide)
%chain.sin.negative = call float @recipe.state.neg(float %chain.sin)
%chain.sin.signed = select i1 %reverse, float %chain.sin.negative, float %chain.sin
%chain.sin.signed.negative = call float @recipe.state.neg(float %chain.sin.signed)
%chain.sin.term = select i1 %upper, float %chain.sin.signed, float %chain.sin.signed.negative
%chain.other.part = call float @recipe.state.mul(float %other, float %chain.sin.term)
%chain.lower = call float @recipe.state.madd(float %chain.other.part, float %value, float %chain.cos)
%chain.upper.cos = call float @recipe.state.mul(float %value, float %chain.cos)
%chain.upper = call float @recipe.state.madd(float %chain.upper.cos, float %other, float %chain.sin.signed)
%rotated.chain = select i1 %upper, float %chain.upper, float %chain.lower
br label %finish.direct
finish.direct:
%rotated.value = phi float [ %rotated.direct, %entry ], [ %rotated.chain, %chain.done ]
ret float %rotated.value
}
define internal void @rope_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %context, i1 %sectioned, i64 %p, i32 %channels, i32 %length,
	i32 %head.width, i32 %dims, i32 %rotated, i8 %base, i8 %yarn.mscale, i8 %yarn.factor, i8 %angle.chain, i8 %yarn.low, i8 %yarn.high, i1 %has.factors, i1 %reverse, i32 %position.origin ) #1 { entry: %channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %head.width.wide = zext i32 %head.width to i64 %dims.wide = zext i32 %dims to i64 %rotated.wide = zext i32 %rotated to i64 %per.row = mul i64 %channels.wide, %length.wide
%within = urem i64 %p, %per.row %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%local = urem i64 %channel, %head.width.wide %half = udiv i64 %dims.wide, 2
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %p
%value.model = load i8, ptr addrspace(1) %input.ptr, align 1
%value = call float @recipe.decode(i8 %value.model)
%rotates = icmp ult i64 %channel, %rotated.wide %inside = icmp ult i64 %local, %dims.wide %active = and i1 %rotates, %inside
%zero = call float @recipe.state.from.u1(i1 false) %one = call float @recipe.state.from.u1(i1 true)
%base.wide = call float @recipe.decode(i8 %base) %mscale.wide = call float @recipe.decode(i8 %yarn.mscale) %factor.wide = call float @recipe.decode(i8 %yarn.factor) %low.wide = call float @recipe.decode(i8 %yarn.low) %high.wide = call float @recipe.decode(i8 %yarn.high)
%unrotated.model = call i8 @recipe.encode(float %value)
br i1 %active, label %rotate, label %finish rotate: %upper = icmp uge i64 %local, %half
%local.upper = sub i64 %local, %half %index = select i1 %upper, i64 %local.upper, i64 %local
%half.stride = mul i64 %half, %length.wide %partner.up = add i64 %p, %half.stride %partner.down = sub i64 %p, %half.stride
%partner = select i1 %upper, i64 %partner.down, i64 %partner.up
%partner.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %partner
%other.model = load i8, ptr addrspace(1) %partner.ptr, align 1
%other = call float @recipe.decode(i8 %other.model)
%two.index = mul i64 %index, 2 %two.index.i32 = trunc i64 %two.index to i32 %index.i32 = trunc i64 %index to i32 %two.index.value = call float @recipe.state.from.u32(i32 %two.index.i32) %index.value = call float @recipe.state.from.u32(i32 %index.i32)
br i1 %has.factors, label %factor.load, label %factor.one
factor.load:
%factor.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %index
%factor.model = load i8, ptr addrspace(1) %factor.ptr, align 1
%factor.loaded = call float @recipe.decode(i8 %factor.model)
br label %factor.ready
factor.one:
br label %factor.ready
factor.ready:
%frequency.factor = phi float [ %factor.loaded, %factor.load ], [ %one, %factor.one ]
%position.i32 = call i32 @rope_position(ptr addrspace(1) %context, i1 %sectioned, i64 %p, i64 %per.row, i64 %length.wide, i64 %position, i32 %index.i32, i32 %position.origin)
%chain.zero = call i8 @recipe.from.u32(i32 0)
%chain.on = call i1 @recipe.ogt(i8 %angle.chain, i8 %chain.zero)
%rotated.value = call float @rope_pair(float %value, float %other, float %base.wide, float %mscale.wide, float %factor.wide, float %low.wide, float %high.wide, float %frequency.factor, i64 %index, i32 %dims, i32 %position.i32, i1 %upper, i1 %reverse, i1 %chain.on)
br label %finish.direct
finish.direct:
%rotated.model = call i8 @recipe.encode(float %rotated.value) br label %finish finish:
%result = phi i8 [ %unrotated.model, %entry ], [ %rotated.model, %finish.direct ]
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p
br i1 %reverse, label %accumulate, label %assign accumulate: %prior.model = load i8, ptr addrspace(1) %output.ptr, align 1
%prior = call float @recipe.decode(i8 %prior.model) %result.wide = call float @recipe.decode(i8 %result) %sum = call float @recipe.state.add(float %prior, float %result.wide) %sum.model = call i8 @recipe.encode(float %sum) store i8 %sum.model, ptr addrspace(1) %output.ptr, align 1 ret void
assign: store i8 %result, ptr addrspace(1) %output.ptr, align 1 ret void }
; The adjoint twin: the same angles and coefficients, read from a state-typed delta
; and added into a state-typed previous, so nothing on the adjoint side is rounded.
define internal void @rope_body_adjoint( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %context, i1 %sectioned, i64 %p, i32 %channels, i32 %length,
 i32 %head.width, i32 %dims, i32 %rotated, i8 %base, i8 %yarn.mscale, i8 %yarn.factor, i8 %angle.chain, i8 %yarn.low, i8 %yarn.high, i1 %has.factors, i1 %reverse ) #1 { entry: %channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %head.width.wide = zext i32 %head.width to i64 %dims.wide = zext i32 %dims to i64 %rotated.wide = zext i32 %rotated to i64 %per.row = mul i64 %channels.wide, %length.wide
%within = urem i64 %p, %per.row %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%local = urem i64 %channel, %head.width.wide %half = udiv i64 %dims.wide, 2
%input.ptr = getelementptr inbounds float, ptr addrspace(1) %input, i64 %p
%value = load float, ptr addrspace(1) %input.ptr, align 4
%rotates = icmp ult i64 %channel, %rotated.wide %inside = icmp ult i64 %local, %dims.wide %active = and i1 %rotates, %inside
%zero = call float @recipe.state.from.u1(i1 false) %one = call float @recipe.state.from.u1(i1 true)
%base.wide = call float @recipe.decode(i8 %base) %mscale.wide = call float @recipe.decode(i8 %yarn.mscale) %factor.wide = call float @recipe.decode(i8 %yarn.factor) %low.wide = call float @recipe.decode(i8 %yarn.low) %high.wide = call float @recipe.decode(i8 %yarn.high)
br i1 %active, label %rotate, label %finish rotate: %upper = icmp uge i64 %local, %half
%local.upper = sub i64 %local, %half %index = select i1 %upper, i64 %local.upper, i64 %local
%half.stride = mul i64 %half, %length.wide %partner.up = add i64 %p, %half.stride %partner.down = sub i64 %p, %half.stride
%partner = select i1 %upper, i64 %partner.down, i64 %partner.up
%partner.ptr = getelementptr inbounds float, ptr addrspace(1) %input, i64 %partner
%other = load float, ptr addrspace(1) %partner.ptr, align 4
%two.index = mul i64 %index, 2 %two.index.i32 = trunc i64 %two.index to i32 %index.i32 = trunc i64 %index to i32 %two.index.value = call float @recipe.state.from.u32(i32 %two.index.i32) %index.value = call float @recipe.state.from.u32(i32 %index.i32)
br i1 %has.factors, label %factor.load, label %factor.one
factor.load:
%factor.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %index
%factor.model = load i8, ptr addrspace(1) %factor.ptr, align 1
%factor.loaded = call float @recipe.decode(i8 %factor.model)
br label %factor.ready
factor.one:
br label %factor.ready
factor.ready:
%frequency.factor = phi float [ %factor.loaded, %factor.load ], [ %one, %factor.one ]
%position.i32 = call i32 @rope_position(ptr addrspace(1) %context, i1 %sectioned, i64 %p, i64 %per.row, i64 %length.wide, i64 %position, i32 %index.i32, i32 0)
%chain.zero = call i8 @recipe.from.u32(i32 0)
%chain.on = call i1 @recipe.ogt(i8 %angle.chain, i8 %chain.zero)
%rotated.value = call float @rope_pair(float %value, float %other, float %base.wide, float %mscale.wide, float %factor.wide, float %low.wide, float %high.wide, float %frequency.factor, i64 %index, i32 %dims, i32 %position.i32, i1 %upper, i1 %reverse, i1 %chain.on)
br label %finish.direct
finish.direct:
br label %finish finish:
%result = phi float [ %value, %entry ], [ %rotated.value, %finish.direct ]
%output.ptr = getelementptr inbounds float, ptr addrspace(1) %output, i64 %p
br i1 %reverse, label %accumulate, label %assign accumulate: %prior = load float, ptr addrspace(1) %output.ptr, align 4
%sum = call float @recipe.state.add(float %prior, float %result) store float %sum, ptr addrspace(1) %output.ptr, align 4 ret void
assign: store float %result, ptr addrspace(1) %output.ptr, align 4 ret void }
; Hyper-connection stream bodies. A stream row holds %lanes copies of
; %channels channels, lane l at channels [l * channels, (l + 1) * channels).
; Reverse bodies add into their adjoints; every element belongs to one thread.
; Output element %p of the widened batch copies the input channel of its lane.
define internal void @expand_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %lanes ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %per.row %within = urem i64 %p, %per.row
%lane.channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %channel = urem i64 %lane.channel, %channels.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %index.row = add i64 %row.base, %channel.base %index = add i64 %index.row, %position
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %index %value = load i8, ptr addrspace(1) %input.ptr, align 1
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %value, ptr addrspace(1) %output.ptr, align 1 ret void }
; Source element %p of the narrow batch sums the adjoints of its lanes in lane order.
define internal void @expand_reverse_body( ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow
%state.zero = call float @recipe.state.from.u1(i1 false)
%row.base = mul i64 %row, %per.row %base = add i64 %row.base, %within br label %loop loop:
%lane = phi i64 [ 0, %entry ], [ %lane.next, %step ] %sum = phi float [ %state.zero, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %lane, %lanes.wide
br i1 %more, label %step, label %done step: %lane.offset = mul i64 %lane, %narrow %index = add i64 %base, %lane.offset
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %index %value = load float, ptr addrspace(1) %delta.ptr, align 4
%sum.next = call float @recipe.state.add(float %sum, float %value) %lane.next = add i64 %lane, 1 br label %loop done:
%adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %p %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%total = call float @recipe.state.add(float %prior, float %sum) store float %total, ptr addrspace(1) %adjoint.ptr, align 4 ret void }
; Output element %p of the narrow batch is the gate-weighted mean of its lanes: the sum in lane order times 1 / lanes; without a gate every lane weighs one.
define internal void @read_forward_body( ptr addrspace(1) %stream, ptr addrspace(1) %gate, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated, i32 %pick ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow
%row.base = mul i64 %row, %per.row %base = add i64 %row.base, %within
%picked = icmp ne i32 %pick, 0 br i1 %picked, label %pick.lane, label %init
pick.lane: %pick.index = sub i32 %pick, 1 %pick.wide = zext i32 %pick.index to i64
%pick.offset = mul i64 %pick.wide, %narrow %pick.at = add i64 %base, %pick.offset
%pick.ptr = getelementptr inbounds i8, ptr addrspace(1) %stream, i64 %pick.at %pick.value = load i8, ptr addrspace(1) %pick.ptr, align 1
%pick.output = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %pick.value, ptr addrspace(1) %pick.output, align 1 ret void
init: %lanes.value = call i8 @recipe.from.u32(i32 %lanes) %scale = call i8 @recipe.div(i8 56, i8 %lanes.value) br label %loop
loop: %lane = phi i64 [ 0, %init ], [ %lane.next, %step ] %sum = phi i8 [ 0, %init ], [ %sum.next, %step ] %more = icmp ult i64 %lane, %lanes.wide
br i1 %more, label %step, label %done step: %lane.offset = mul i64 %lane, %narrow %index = add i64 %base, %lane.offset
%stream.ptr = getelementptr inbounds i8, ptr addrspace(1) %stream, i64 %index %value = load i8, ptr addrspace(1) %stream.ptr, align 1
%gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %gate, i64 %index %gate.loaded = load i8, ptr addrspace(1) %gate.ptr, align 1
%weight = select i1 %gated, i8 %gate.loaded, i8 56 %product = call i8 @recipe.mul(i8 %weight, i8 %value)
%sum.next = call i8 @recipe.add(i8 %sum, i8 %product) %lane.next = add i64 %lane, 1 br label %loop done:
%mean = call i8 @recipe.mul(i8 %sum, i8 %scale)
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %mean, ptr addrspace(1) %output.ptr, align 1 ret void }
; Stream element %p receives gate * dh / lanes, and its gate receives stream * dh / lanes.
define internal void @read_reverse_body( ptr addrspace(1) %stream, ptr addrspace(1) %gate, ptr addrspace(1) %delta, ptr addrspace(1) %stream.adjoint, ptr addrspace(1) %gate.adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %per.row %within = urem i64 %p, %per.row
%lane.channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %channel = urem i64 %lane.channel, %channels.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %h.row = add i64 %row.base, %channel.base %h = add i64 %h.row, %position
%state.one = call float @recipe.state.from.u1(i1 true) %lanes.value = call float @recipe.state.from.u32(i32 %lanes) %scale = call float @recipe.state.div(float %state.one, float %lanes.value)
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %h %dh.loaded = load float, ptr addrspace(1) %delta.ptr, align 4
%dh = call float @recipe.state.mul(float %dh.loaded, float %scale)
%gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %gate, i64 %p %gate.loaded = load i8, ptr addrspace(1) %gate.ptr, align 1 %gate.wide = call float @recipe.decode(i8 %gate.loaded)
%weight = select i1 %gated, float %gate.wide, float %state.one %stream.term = call float @recipe.state.mul(float %weight, float %dh)
%stream.adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %stream.adjoint, i64 %p %stream.prior = load float, ptr addrspace(1) %stream.adjoint.ptr, align 4
%stream.sum = call float @recipe.state.add(float %stream.prior, float %stream.term) store float %stream.sum, ptr addrspace(1) %stream.adjoint.ptr, align 4
br i1 %gated, label %gate.pass, label %exit gate.pass:
%stream.ptr = getelementptr inbounds i8, ptr addrspace(1) %stream, i64 %p %value = load i8, ptr addrspace(1) %stream.ptr, align 1 %value.wide = call float @recipe.decode(i8 %value)
%gate.term = call float @recipe.state.mul(float %value.wide, float %dh)
%gate.adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %gate.adjoint, i64 %p %gate.prior = load float, ptr addrspace(1) %gate.adjoint.ptr, align 4
%gate.sum = call float @recipe.state.add(float %gate.prior, float %gate.term) store float %gate.sum, ptr addrspace(1) %gate.adjoint.ptr, align 4 br label %exit exit: ret void }
; Output element %p of the widened batch is its lane's write gate times the branch output channel.
define internal void @outer_forward_body( ptr addrspace(1) %branch, ptr addrspace(1) %gate, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated, i32 %pick ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %per.row %within = urem i64 %p, %per.row
%lane.channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %channel = urem i64 %lane.channel, %channels.wide %lane = udiv i64 %lane.channel, %channels.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %y.row = add i64 %row.base, %channel.base %y = add i64 %y.row, %position
%gate.row = mul i64 %row, %lanes.wide %gate.lane = add i64 %gate.row, %lane %gate.lane.base = mul i64 %gate.lane, %length.wide %g = add i64 %gate.lane.base, %position
%branch.ptr = getelementptr inbounds i8, ptr addrspace(1) %branch, i64 %y %value = load i8, ptr addrspace(1) %branch.ptr, align 1
%gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %gate, i64 %g %gate.loaded = load i8, ptr addrspace(1) %gate.ptr, align 1
%base.weight = select i1 %gated, i8 %gate.loaded, i8 56
%pick.index = sub i32 %pick, 1 %pick.wide = zext i32 %pick.index to i64 %pick.lane = icmp eq i64 %lane, %pick.wide %all.lanes = icmp eq i32 %pick, 0 %selected = or i1 %all.lanes, %pick.lane
%weight = select i1 %selected, i8 %base.weight, i8 0 %product = call i8 @recipe.mul(i8 %weight, i8 %value)
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %product, ptr addrspace(1) %output.ptr, align 1 ret void }
; Branch element %p sums gate * adjoint over its lanes in lane order.
define internal void @outer_reverse_branch_body( ptr addrspace(1) %gate, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %position = urem i64 %within, %length.wide
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%row.base = mul i64 %row, %per.row %base = add i64 %row.base, %within %gate.row = mul i64 %row, %lanes.wide br label %loop loop:
%lane = phi i64 [ 0, %entry ], [ %lane.next, %step ] %sum = phi float [ %state.zero, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %lane, %lanes.wide
br i1 %more, label %step, label %done step: %lane.offset = mul i64 %lane, %narrow %index = add i64 %base, %lane.offset
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %index %value = load float, ptr addrspace(1) %delta.ptr, align 4
%gate.lane = add i64 %gate.row, %lane %gate.lane.base = mul i64 %gate.lane, %length.wide %g = add i64 %gate.lane.base, %position
%gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %gate, i64 %g %gate.loaded = load i8, ptr addrspace(1) %gate.ptr, align 1 %gate.wide = call float @recipe.decode(i8 %gate.loaded)
%weight = select i1 %gated, float %gate.wide, float %state.one %product = call float @recipe.state.mul(float %weight, float %value)
%sum.next = call float @recipe.state.add(float %sum, float %product) %lane.next = add i64 %lane, 1 br label %loop done:
%adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %p %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%total = call float @recipe.state.add(float %prior, float %sum) store float %total, ptr addrspace(1) %adjoint.ptr, align 4 ret void }
; Gate element %p (one per row, lane, and position) sums branch * adjoint over its channels in channel order.
define internal void @outer_reverse_gate_body( ptr addrspace(1) %branch, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %gates.row = mul i64 %lanes.wide, %length.wide %row = udiv i64 %p, %gates.row %within = urem i64 %p, %gates.row
%lane = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %row.base = mul i64 %row, %per.row %lane.base = mul i64 %lane, %narrow
%delta.base.row = add i64 %row.base, %lane.base %delta.base = add i64 %delta.base.row, %position %branch.row = mul i64 %row, %narrow %branch.base = add i64 %branch.row, %position
%state.zero = call float @recipe.state.from.u1(i1 false)
br label %loop loop: %channel = phi i64 [ 0, %entry ], [ %channel.next, %step ] %sum = phi float [ %state.zero, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %channel, %channels.wide
br i1 %more, label %step, label %done step: %channel.offset = mul i64 %channel, %length.wide %delta.index = add i64 %delta.base, %channel.offset %branch.index = add i64 %branch.base, %channel.offset
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %delta.index %value = load float, ptr addrspace(1) %delta.ptr, align 4
%branch.ptr = getelementptr inbounds i8, ptr addrspace(1) %branch, i64 %branch.index %y = load i8, ptr addrspace(1) %branch.ptr, align 1 %y.wide = call float @recipe.decode(i8 %y)
%product = call float @recipe.state.mul(float %y.wide, float %value) %sum.next = call float @recipe.state.add(float %sum, float %product) %channel.next = add i64 %channel, 1 br label %loop done:
%adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %p %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%total = call float @recipe.state.add(float %prior, float %sum) store float %total, ptr addrspace(1) %adjoint.ptr, align 4 ret void }
; Channel-group sum. Output element %p of a [rows][groups][length] batch sums the
; %width channels of its group at its position, in channel order.
define internal void @fold_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %output, i64 %p, i32 %groups, i32 %width, i32 %length ) #1 { entry:
%groups.wide = zext i32 %groups to i64 %width.wide = zext i32 %width to i64 %length.wide = zext i32 %length to i64
%narrow = mul i64 %groups.wide, %length.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %group = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%channels = mul i64 %groups.wide, %width.wide %wide = mul i64 %channels, %length.wide %row.base = mul i64 %row, %wide %first = mul i64 %group, %width.wide %first.base = mul i64 %first, %length.wide
%base.row = add i64 %row.base, %first.base %base = add i64 %base.row, %position br label %loop loop:
%c = phi i64 [ 0, %entry ], [ %c.next, %step ] %sum = phi i8 [ 0, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %c, %width.wide br i1 %more, label %step, label %done
step: %offset = mul i64 %c, %length.wide %index = add i64 %base, %offset %input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %index %value = load i8, ptr addrspace(1) %input.ptr, align 1
%sum.next = call i8 @recipe.add(i8 %sum, i8 %value) %c.next = add i64 %c, 1 br label %loop
done: %output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %sum, ptr addrspace(1) %output.ptr, align 1 ret void }
; Input element %p receives the adjoint of its group at its position.
define internal void @fold_reverse_body( ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %groups, i32 %width, i32 %length ) #1 { entry:
%groups.wide = zext i32 %groups to i64 %width.wide = zext i32 %width to i64 %length.wide = zext i32 %length to i64
%channels = mul i64 %groups.wide, %width.wide %wide = mul i64 %channels, %length.wide %row = udiv i64 %p, %wide %within = urem i64 %p, %wide %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%group = udiv i64 %channel, %width.wide %narrow = mul i64 %groups.wide, %length.wide %row.base = mul i64 %row, %narrow %group.base = mul i64 %group, %length.wide %index.row = add i64 %row.base, %group.base %index = add i64 %index.row, %position
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %index %value = load float, ptr addrspace(1) %delta.ptr, align 4
%adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %p %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%total = call float @recipe.state.add(float %prior, float %value) store float %total, ptr addrspace(1) %adjoint.ptr, align 4 ret void }
; Causal depthwise convolution. Output element %p of a [rows][channels][length]
; batch sums %kernel positions of its own channel, tap j reading position
; t - (kernel - 1 - j) * dilation, with positions before the start reading zero.
define internal void @dconv_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %context, i64 %p, i32 %channels, i32 %length, i32 %kernel, i32 %dilation, i32 %decode, i32 %origin, i32 %history ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %kernel.wide = zext i32 %kernel to i64 %dilation.wide = zext i32 %dilation to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %base = add i64 %row.base, %channel.base %tap.base = mul i64 %channel, %kernel.wide
%origin.wide = zext i32 %origin to i64
%history.enabled = icmp ne i32 %history, 0
%history.safe = select i1 %history.enabled, i32 %history, i32 1
%history.wide = zext i32 %history.safe to i64
%history.row = mul i64 %row, %channels.wide
%history.channel = add i64 %history.row, %channel
%history.base = mul i64 %history.channel, %history.wide
%reach = sub i64 %kernel.wide, 1 br label %loop loop: %tap = phi i64 [ 0, %entry ], [ %tap.next, %step ] %sum = phi i8 [ 0, %entry ], [ %sum.next, %step ]
%more = icmp ult i64 %tap, %kernel.wide br i1 %more, label %step, label %done step: %back = sub i64 %reach, %tap %back.scaled = mul i64 %back, %dilation.wide %source = sub i64 %position, %back.scaled
%global.source = add i64 %origin.wide, %source
%valid = icmp sge i64 %global.source, 0
%local = icmp sge i64 %source, 0
%source.clamped = select i1 %local, i64 %source, i64 0
%index = add i64 %base, %source.clamped
%input.ptr = getelementptr i8, ptr addrspace(1) %input, i64 %index
%global.clamped = select i1 %valid, i64 %global.source, i64 0
%history.slot = urem i64 %global.clamped, %history.wide
%history.index = add i64 %history.base, %history.slot
%history.ptr = getelementptr i8, ptr addrspace(1) %context, i64 %history.index
%read.ptr = select i1 %local, ptr addrspace(1) %input.ptr, ptr addrspace(1) %history.ptr
%valid.ptr = select i1 %valid, ptr addrspace(1) %read.ptr, ptr addrspace(1) %input.ptr
%loaded = load i8, ptr addrspace(1) %valid.ptr, align 1
%value = select i1 %valid, i8 %loaded, i8 0
%tap.index = add i64 %tap.base, %tap %weight = call i8 @recipe.model.weight(ptr addrspace(1) %weights, i64 %tap.index, i32 %decode)
%product = call i8 @recipe.mul(i8 %weight, i8 %value) %sum.next = call i8 @recipe.add(i8 %sum, i8 %product) %tap.next = add i64 %tap, 1 br label %loop
done: %output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %sum, ptr addrspace(1) %output.ptr, align 1
br i1 %history.enabled, label %history.save, label %exit
history.save:
%global.position = add i64 %origin.wide, %position
%save.slot = urem i64 %global.position, %history.wide
%save.index = add i64 %history.base, %save.slot
%save.ptr = getelementptr i8, ptr addrspace(1) %context, i64 %save.index
%current.ptr = getelementptr i8, ptr addrspace(1) %input, i64 %p
%current = load i8, ptr addrspace(1) %current.ptr, align 1
store i8 %current, ptr addrspace(1) %save.ptr, align 1
br label %exit
exit: ret void }
; Input element %p receives tap j times the adjoint at position t + (kernel - 1 - j) * dilation while that position exists.
define internal void @dconv_reverse_input_body( ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %kernel, i32 %dilation ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %kernel.wide = zext i32 %kernel to i64 %dilation.wide = zext i32 %dilation to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %base = add i64 %row.base, %channel.base %tap.base = mul i64 %channel, %kernel.wide
%state.zero = call float @recipe.state.from.u1(i1 false)
%reach = sub i64 %kernel.wide, 1 br label %loop loop: %tap = phi i64 [ 0, %entry ], [ %tap.next, %step ] %sum = phi float [ %state.zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i64 %tap, %kernel.wide br i1 %more, label %step, label %done step: %ahead.taps = sub i64 %reach, %tap %ahead = mul i64 %ahead.taps, %dilation.wide %target = add i64 %position, %ahead
%valid = icmp ult i64 %target, %length.wide %target.clamped = select i1 %valid, i64 %target, i64 0 %index = add i64 %base, %target.clamped
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %index %loaded = load float, ptr addrspace(1) %delta.ptr, align 4 %value = select i1 %valid, float %loaded, float %state.zero
%tap.index = add i64 %tap.base, %tap %weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %tap.index %weight = load i8, ptr addrspace(1) %weight.ptr, align 1 %weight.wide = call float @recipe.decode(i8 %weight)
%product = call float @recipe.state.mul(float %weight.wide, float %value) %sum.next = call float @recipe.state.add(float %sum, float %product) %tap.next = add i64 %tap, 1 br label %loop
done: %adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %p %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%total = call float @recipe.state.add(float %prior, float %sum) store float %total, ptr addrspace(1) %adjoint.ptr, align 4 ret void }
; Tap %p (channel * kernel + j) sums input * adjoint over every row and position in order and writes its gradient.
define internal void @dconv_reverse_weight_body( ptr addrspace(1) %input, ptr addrspace(1) %delta, ptr addrspace(1) %gradient, i64 %p, i32 %rows, i32 %channels, i32 %length, i32 %kernel, i32 %dilation, i32 %offset ) #1 { entry:
%kernel.wide = zext i32 %kernel to i64 %channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %dilation.wide = zext i32 %dilation to i64 %rows.wide = zext i32 %rows to i64 %offset.wide = zext i32 %offset to i64
%channel = udiv i64 %p, %kernel.wide %tap = urem i64 %p, %kernel.wide %narrow = mul i64 %channels.wide, %length.wide %channel.base = mul i64 %channel, %length.wide %reach = sub i64 %kernel.wide, 1 %shift.taps = sub i64 %reach, %tap %shift = mul i64 %shift.taps, %dilation.wide
%state.zero = call float @recipe.state.from.u1(i1 false)
%count = mul i64 %rows.wide, %length.wide br label %loop loop: %step.index = phi i64 [ 0, %entry ], [ %step.next, %step ] %sum = phi float [ %state.zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i64 %step.index, %count br i1 %more, label %step, label %done step: %row = udiv i64 %step.index, %length.wide %position = urem i64 %step.index, %length.wide
%valid = icmp uge i64 %position, %shift %source = sub i64 %position, %shift %source.clamped = select i1 %valid, i64 %source, i64 0
%row.base = mul i64 %row, %narrow %base = add i64 %row.base, %channel.base %input.index = add i64 %base, %source.clamped %delta.index = add i64 %base, %position
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %input.index %loaded = load i8, ptr addrspace(1) %input.ptr, align 1 %loaded.wide = call float @recipe.decode(i8 %loaded) %value = select i1 %valid, float %loaded.wide, float %state.zero
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %delta.index %incoming = load float, ptr addrspace(1) %delta.ptr, align 4
%product = call float @recipe.state.mul(float %value, float %incoming) %sum.next = call float @recipe.state.add(float %sum, float %product) %step.next = add i64 %step.index, 1 br label %loop
done: %gradient.index = add i64 %offset.wide, %p %gradient.ptr = getelementptr inbounds float, ptr addrspace(1) %gradient, i64 %gradient.index store float %sum, ptr addrspace(1) %gradient.ptr, align 4 ret void }
; log(1 + exp(x)) taken on the negative side so a large x cannot overflow.
define internal float @delta_softplus(float %x) #1 { entry:
%state.one = call float @recipe.state.from.u1(i1 true)
%magnitude = call float @recipe.state.abs(float %x)
%negative = call float @recipe.state.neg(float %magnitude) %exponential = call float @recipe.state.exp(float %negative)
%shifted = call float @recipe.state.add(float %state.one, float %exponential) %tail = call float @recipe.state.log(float %shifted)
%softplus.zero = call float @recipe.state.from.u1(i1 false) %positive = call i1 @recipe.state.ogt(float %x, float %softplus.zero) %linear = select i1 %positive, float %x, float %softplus.zero
%value = call float @recipe.state.add(float %linear, float %tail) ret float %value }
; Prepare the scalars shared by all value columns of one head and position.
define internal <3 x float> @delta_parameters(ptr addrspace(1) %gates, i64 %a.base, i64 %b.base, i32 %kwidth, i32 %time, float %decay.scale) #1 { entry:
%state.one = call float @recipe.state.from.u1(i1 true)

%query.width = call float @recipe.state.from.u32(i32 %kwidth)
%query.root = call float @recipe.state.sqrt(float %query.width)
%query.scale = call float @recipe.state.div(float %state.one, float %query.root)
%time.wide = zext i32 %time to i64
%decay.index = add i64 %a.base, %time.wide
%decay.pointer = getelementptr inbounds i8, ptr addrspace(1) %gates, i64 %decay.index
%decay.input.model = load i8, ptr addrspace(1) %decay.pointer, align 1
%decay.input = call float @recipe.decode(i8 %decay.input.model) %softplus = call float @delta_softplus(float %decay.input)
%exponent = call float @recipe.state.mul(float %softplus, float %decay.scale) %negated = call float @recipe.state.neg(float %exponent)
%decay = call float @recipe.state.exp(float %negated) %write.index = add i64 %b.base, %time.wide
%write.pointer = getelementptr inbounds i8, ptr addrspace(1) %gates, i64 %write.index
%write.input.model = load i8, ptr addrspace(1) %write.pointer, align 1
%write.input = call float @recipe.decode(i8 %write.input.model) %write = call float @recipe.state.sigmoid(float %write.input)
%result.q = insertelement <3 x float> poison, float %query.scale, i32 0
%result.d = insertelement <3 x float> %result.q, float %decay, i32 1
%result.w = insertelement <3 x float> %result.d, float %write, i32 2
ret <3 x float> %result.w }
; One head prepares its gates once for the live window. Columns read these
; scalars after the existing compiled unit's grid barrier.
define internal void @delta_prepare_body(ptr addrspace(1) %gates, ptr addrspace(1) %weights, ptr addrspace(1) %prepared,
i64 %p, i32 %heads, i32 %kwidth, i32 %length, i32 %begin, i32 %end, i32 %decode) #3 { entry:
%heads.wide = zext i32 %heads to i64 %length.wide = zext i32 %length to i64
%row = udiv i64 %p, %heads.wide %head = urem i64 %p, %heads.wide
%stream = mul i64 %heads.wide, %length.wide %row.stream = mul i64 %row, %stream %row.base = mul i64 %row.stream, 2
%head.base = mul i64 %head, %length.wide %a.base = add i64 %row.base, %head.base %b.base = add i64 %a.base, %stream
%decay.parameter = call i8 @recipe.model.weight(ptr addrspace(1) %weights, i64 %head, i32 %decode)
%decay.parameter.wide = call float @recipe.decode(i8 %decay.parameter)
%decay.scale = call float @recipe.state.exp(float %decay.parameter.wide)
%pair.base = mul i64 %p, %length.wide
br label %loop
loop: %time = phi i32 [ %begin, %entry ], [ %next, %step ]
%more = icmp ult i32 %time, %end br i1 %more, label %step, label %exit
step:
%parameters = call <3 x float> @delta_parameters(ptr addrspace(1) %gates, i64 %a.base, i64 %b.base, i32 %kwidth, i32 %time, float %decay.scale)
%q = extractelement <3 x float> %parameters, i32 0
%d = extractelement <3 x float> %parameters, i32 1
%w = extractelement <3 x float> %parameters, i32 2
%time.wide = zext i32 %time to i64 %position = add i64 %pair.base, %time.wide %base = mul i64 %position, 3
%q.ptr = getelementptr float, ptr addrspace(1) %prepared, i64 %base
%d.ptr = getelementptr float, ptr addrspace(1) %q.ptr, i64 1
%w.ptr = getelementptr float, ptr addrspace(1) %q.ptr, i64 2
store float %q, ptr addrspace(1) %q.ptr, align 4
store float %d, ptr addrspace(1) %d.ptr, align 4
store float %w, ptr addrspace(1) %w.ptr, align 4
%next = add i32 %time, 1 br label %loop
exit: ret void }
; Each column owns its state cells, preserving the serial key-reduction order.
define internal void @delta_step( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base, i64 %work.base,
i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, float %decay.scale, i1 %store,
i32 %column.begin, i32 %column.stride, ptr addrspace(1) %prepared ) #3 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
%time.wide = zext i32 %time to i64 %kwidth.wide = zext i32 %kwidth to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64
%cached = icmp ne ptr addrspace(1) %prepared, null
br i1 %cached, label %load, label %compute
compute:
%parameters = call <3 x float> @delta_parameters(ptr addrspace(1) %gates, i64 %a.base, i64 %b.base, i32 %kwidth, i32 %time, float %decay.scale)
%computed.q = extractelement <3 x float> %parameters, i32 0
%computed.d = extractelement <3 x float> %parameters, i32 1
%computed.w = extractelement <3 x float> %parameters, i32 2
br label %columns
load:
%base = mul i64 %time.wide, 3
%q.ptr = getelementptr float, ptr addrspace(1) %prepared, i64 %base
%d.ptr = getelementptr float, ptr addrspace(1) %q.ptr, i64 1
%w.ptr = getelementptr float, ptr addrspace(1) %q.ptr, i64 2
%loaded.q = load float, ptr addrspace(1) %q.ptr, align 4
%loaded.d = load float, ptr addrspace(1) %d.ptr, align 4
%loaded.w = load float, ptr addrspace(1) %w.ptr, align 4
br label %columns
columns:
%query.scale = phi float [ %computed.q, %compute ], [ %loaded.q, %load ]
%decay = phi float [ %computed.d, %compute ], [ %loaded.d, %load ]
%write = phi float [ %computed.w, %compute ], [ %loaded.w, %load ]
br label %column.loop
column.loop: %column = phi i32 [ %column.begin, %columns ], [ %column.next, %column.done ] %column.wide = zext i32 %column to i64 %column.more = icmp ult i32 %column, %vwidth
br i1 %column.more, label %read.loop, label %exit
read.loop: %read.i = phi i32 [ 0, %column.loop ], [ %read.next, %read.step ]
%read.sum = phi float [ %state.zero, %column.loop ], [ %read.sum.next, %read.step ]
%read.more = icmp ult i32 %read.i, %kwidth br i1 %read.more, label %read.step, label %read.done
read.step: %read.i.wide = zext i32 %read.i to i64 %read.row = mul i64 %read.i.wide, %vwidth.wide %read.cell = add i64 %read.row, %column.wide %read.index = add i64 %work.base, %read.cell
%read.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %read.index
%read.state = load float, ptr addrspace(1) %read.pointer, align 4
; Decay the old state before predicting the value removed by the delta update.
%read.decayed = call float @recipe.state.mul(float %decay, float %read.state)
store float %read.decayed, ptr addrspace(1) %read.pointer, align 4
%read.offset.row = mul i64 %read.i.wide, %length.wide %read.offset = add i64 %read.offset.row, %time.wide %read.key.index = add i64 %k.base, %read.offset
%read.key.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %read.key.index
%read.key.model = load i8, ptr addrspace(1) %read.key.pointer, align 1
%read.key = call float @recipe.decode(i8 %read.key.model)
%read.product = call float @recipe.state.mul(float %read.key, float %read.decayed)
%read.sum.next = call float @recipe.state.add(float %read.sum, float %read.product) %read.next = add nuw i32 %read.i, 1 br label %read.loop
read.done: %value.row = mul i64 %column.wide, %length.wide %value.offset = add i64 %value.row, %time.wide %value.index = add i64 %v.base, %value.offset
%value.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %value.index
%value.model = load i8, ptr addrspace(1) %value.pointer, align 1
%value = call float @recipe.decode(i8 %value.model)
%error = call float @recipe.state.sub(float %value, float %read.sum) %write.error = call float @recipe.state.mul(float %write, float %error)
br label %write.loop
write.loop: %write.i = phi i32 [ 0, %read.done ], [ %write.next, %write.step ]
%write.sum = phi float [ %state.zero, %read.done ], [ %write.sum.next, %write.step ]
%write.more = icmp ult i32 %write.i, %kwidth br i1 %write.more, label %write.step, label %write.done
write.step: %write.i.wide = zext i32 %write.i to i64 %write.row = mul i64 %write.i.wide, %vwidth.wide %write.cell = add i64 %write.row, %column.wide %write.cell.index = add i64 %work.base, %write.cell
%write.state.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %write.cell.index
%write.state = load float, ptr addrspace(1) %write.state.pointer, align 4
%write.offset.row = mul i64 %write.i.wide, %length.wide %write.offset = add i64 %write.offset.row, %time.wide
%write.key.index = add i64 %k.base, %write.offset
%write.key.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %write.key.index
%write.key.model = load i8, ptr addrspace(1) %write.key.pointer, align 1
%write.key = call float @recipe.decode(i8 %write.key.model)
%write.term = call float @recipe.state.mul(float %write.key, float %write.error)
%write.state.next = call float @recipe.state.add(float %write.state, float %write.term)
store float %write.state.next, ptr addrspace(1) %write.state.pointer, align 4
%write.query.index = add i64 %q.base, %write.offset
%write.query.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %write.query.index
%write.query.model = load i8, ptr addrspace(1) %write.query.pointer, align 1
%write.query = call float @recipe.decode(i8 %write.query.model)
%write.query.scaled = call float @recipe.state.mul(float %write.query, float %query.scale)
%write.output = call float @recipe.state.mul(float %write.query.scaled, float %write.state.next)
%write.sum.next = call float @recipe.state.add(float %write.sum, float %write.output) %write.next = add nuw i32 %write.i, 1 br label %write.loop
write.done: br i1 %store, label %write.store, label %column.done
write.store: %output.index = add i64 %o.base, %value.offset
%output.pointer = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.index
%output.model = call i8 @recipe.encode(float %write.sum)
store i8 %output.model, ptr addrspace(1) %output.pointer, align 1 br label %column.done
column.done: %column.next = add nuw i32 %column, %column.stride br label %column.loop
exit: ret void }
; One row and head of the gated delta rule. The sequence walks in chunks of
; %chunk positions and the carried state is committed at every chunk start, so a
; chunk of one commits each decode step. The chunk never reaches the arithmetic.
define internal void @delta_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %p, i32 %kheads, i32 %kwidth, i32 %vheads, i32 %vwidth, i32 %length, i32 %chunk, i32 %chunks, i32 %pairs, i32 %tiled, i32 %entries, i32 %decode, i32 %begin, i32 %end, i32 %checkpoints, i32 %origin,
i32 %column.begin, i32 %column.stride, ptr addrspace(1) %prepared ) #3 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
%state.one = call float @recipe.state.from.u1(i1 true)

%kheads.wide = zext i32 %kheads to i64 %kwidth.wide = zext i32 %kwidth to i64 %vheads.wide = zext i32 %vheads to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64 %chunk.wide = zext i32 %chunk to i64 %chunks.wide = zext i32 %chunks to i64 %pairs.wide = zext i32 %pairs to i64 %entries.wide = zext i32 %entries to i64
%row = udiv i64 %p, %vheads.wide %head = urem i64 %p, %vheads.wide %state = mul i64 %kwidth.wide, %vwidth.wide
%state.i32 = trunc i64 %state to i32 %committing = icmp ne i32 %entries, 0 %commit.count = select i1 %committing, i32 %state.i32, i32 0
%kchannels = mul i64 %kheads.wide, %kwidth.wide %kstream = mul i64 %kchannels, %length.wide
%vchannels = mul i64 %vheads.wide, %vwidth.wide %stream = mul i64 %vchannels, %length.wide
%kplanes = mul i64 %kstream, 2 %row.stride = add i64 %kplanes, %stream
%input.row = mul i64 %row, %row.stride %group = udiv i32 %vheads, %kheads %group.wide = zext i32 %group to i64
%khead.grouped = udiv i64 %head, %group.wide %khead.tiled = urem i64 %head, %kheads.wide %is.tiled = icmp ne i32 %tiled, 0
%khead = select i1 %is.tiled, i64 %khead.tiled, i64 %khead.grouped
%khead.base = mul i64 %khead, %kwidth.wide %khead.offset = mul i64 %khead.base, %length.wide
%head.base = mul i64 %head, %vwidth.wide %head.offset = mul i64 %head.base, %length.wide
%q.base = add i64 %input.row, %khead.offset %k.base = add i64 %q.base, %kstream
%value.plane = add i64 %input.row, %kplanes %v.base = add i64 %value.plane, %head.offset
%output.row = mul i64 %row, %stream %o.base = add i64 %output.row, %head.offset
%gate.stream = mul i64 %vheads.wide, %length.wide %gate.row = mul i64 %row, %gate.stream %gate.pair = mul i64 %gate.row, 2
%head.length = mul i64 %head, %length.wide %a.base = add i64 %gate.pair, %head.length %b.base = add i64 %a.base, %gate.stream
%entry.span = mul i64 %entries.wide, %state %entry.base = mul i64 %p, %entry.span
%work.region = mul i64 %pairs.wide, %entry.span %work.offset = mul i64 %p, %state %work.base = add i64 %work.region, %work.offset
%cached = icmp ne ptr addrspace(1) %prepared, null
br i1 %cached, label %initialize.entry, label %scale
scale:
%decay.parameter = call i8 @recipe.model.weight(ptr addrspace(1) %weights, i64 %head, i32 %decode) %decay.parameter.wide = call float @recipe.decode(i8 %decay.parameter)
%computed.scale = call float @recipe.state.exp(float %decay.parameter.wide)
br label %initialize.entry
initialize.entry:
%decay.scale = phi float [ %state.zero, %entry ], [ %computed.scale, %scale ]
%global.begin = add i32 %origin, %begin
%fresh = icmp eq i32 %global.begin, 0
%checkpointing = icmp ne i32 %checkpoints, 0
%initialize = or i1 %fresh, %checkpointing
%zero.count = select i1 %initialize, i32 %state.i32, i32 0
%checkpoint.divisor = select i1 %checkpointing, i32 %checkpoints, i32 1
%previous = sub i32 %global.begin, 1
%restore.slot = urem i32 %previous, %checkpoint.divisor
%restore.slot.wide = zext i32 %restore.slot to i64
%checkpoint.slots = zext i32 %checkpoints to i64
%checkpoint.region = mul i64 %pairs.wide, %state
%checkpoint.pair = mul i64 %p, %checkpoint.slots
%restore.pair = add i64 %checkpoint.pair, %restore.slot.wide
%restore.offset = mul i64 %restore.pair, %state
%restore.base = add i64 %checkpoint.region, %restore.offset
%first.chunk = udiv i32 %begin, %chunk
br label %zero.loop
zero.loop: %zero.i = phi i32 [ %column.begin, %initialize.entry ], [ %zero.next, %zero.store ] %zero.more = icmp ult i32 %zero.i, %zero.count
br i1 %zero.more, label %zero.step, label %chunk.loop
zero.step: %zero.i.wide = zext i32 %zero.i to i64 %zero.index = add i64 %work.base, %zero.i.wide
%zero.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %zero.index
br i1 %fresh, label %zero.store, label %restore.load
restore.load:
%restore.index = add i64 %restore.base, %zero.i.wide
%restore.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %restore.index
%restored = load float, ptr addrspace(1) %restore.pointer, align 4
br label %zero.store
zero.store:
%initial = phi float [ %state.zero, %zero.step ], [ %restored, %restore.load ]
store float %initial, ptr addrspace(1) %zero.pointer, align 4 %zero.next = add nuw i32 %zero.i, %column.stride br label %zero.loop
chunk.loop: %chunk.index = phi i32 [ %first.chunk, %zero.loop ], [ %chunk.next, %chunk.done ] %chunk.exists = icmp ult i32 %chunk.index, %chunks
%chunk.start = mul i32 %chunk.index, %chunk %chunk.index.wide = zext i32 %chunk.index to i64 %chunk.entry = mul i64 %chunk.index.wide, %state %chunk.entry.base = add i64 %entry.base, %chunk.entry
%chunk.live = icmp ult i32 %chunk.start, %end
%chunk.more = and i1 %chunk.exists, %chunk.live
%chunk.partial = icmp ult i32 %chunk.start, %begin
%chunk.skip = sub i32 %begin, %chunk.start
%offset.start = select i1 %chunk.partial, i32 %chunk.skip, i32 0
%entry.count = select i1 %chunk.partial, i32 0, i32 %commit.count
br i1 %chunk.more, label %commit.loop, label %exit
commit.loop: %commit.i = phi i32 [ %column.begin, %chunk.loop ], [ %commit.next, %commit.step ] %commit.more = icmp ult i32 %commit.i, %entry.count
br i1 %commit.more, label %commit.step, label %time.loop
commit.step: %commit.i.wide = zext i32 %commit.i to i64 %commit.work = add i64 %work.base, %commit.i.wide
%commit.work.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %commit.work
%commit.value = load float, ptr addrspace(1) %commit.work.pointer, align 4 %commit.entry = add i64 %chunk.entry.base, %commit.i.wide
%commit.entry.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %commit.entry
store float %commit.value, ptr addrspace(1) %commit.entry.pointer, align 4 %commit.next = add nuw i32 %commit.i, %column.stride br label %commit.loop
time.loop: %offset = phi i32 [ %offset.start, %commit.loop ], [ %offset.next, %step.done ] %time = add i32 %chunk.start, %offset
%offset.more = icmp ult i32 %offset, %chunk %time.more = icmp ult i32 %time, %end %step.more = and i1 %offset.more, %time.more
br i1 %step.more, label %step, label %chunk.done
step: call void @delta_step( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base, i64 %work.base,
i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, float %decay.scale, i1 true, i32 %column.begin, i32 %column.stride, ptr addrspace(1) %prepared )
%global.time = add i32 %origin, %time
%save.slot = urem i32 %global.time, %checkpoint.divisor
%save.slot.wide = zext i32 %save.slot to i64
%save.pair = add i64 %checkpoint.pair, %save.slot.wide
%save.offset = mul i64 %save.pair, %state
%save.base = add i64 %checkpoint.region, %save.offset
%save.count = select i1 %checkpointing, i32 %state.i32, i32 0
br label %save.loop
save.loop:
%save.i = phi i32 [ %column.begin, %step ], [ %save.next, %save.step ]
%save.more = icmp ult i32 %save.i, %save.count
br i1 %save.more, label %save.step, label %step.done
save.step:
%save.i.wide = zext i32 %save.i to i64
%save.source = add i64 %work.base, %save.i.wide
%save.source.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %save.source
%save.value = load float, ptr addrspace(1) %save.source.pointer, align 4
%save.destination = add i64 %save.base, %save.i.wide
%save.destination.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %save.destination
store float %save.value, ptr addrspace(1) %save.destination.pointer, align 4
%save.next = add nuw i32 %save.i, %column.stride
br label %save.loop
step.done: %offset.next = add nuw i32 %offset, 1 br label %time.loop
chunk.done: %chunk.next = add nuw i32 %chunk.index, 1 br label %chunk.loop
exit: ret void }
; One position of the gated delta rule in reverse. %previous indexes the state
; before the position and the state adjoint at %adjoint.base is carried
; backward. The two head vectors at %vector.base hold the readout error and the
; key adjoint weight in that order. The return value is this position's
; contribution to the decay scale gradient.
define internal float @delta_back( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %context, ptr addrspace(1) %backward, ptr addrspace(1) %delta,
ptr addrspace(1) %input.adjoint, ptr addrspace(1) %gate.adjoint, i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base,
i64 %previous, i64 %adjoint.base, i64 %vector.base, i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, float %decay.scale ) #3 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%query.width = call float @recipe.state.from.u32(i32 %kwidth)
%query.root = call float @recipe.state.sqrt(float %query.width)
%query.scale = call float @recipe.state.div(float %state.one, float %query.root)
%time.wide = zext i32 %time to i64 %kwidth.wide = zext i32 %kwidth to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64
%decay.index = add i64 %a.base, %time.wide
%decay.pointer = getelementptr inbounds i8, ptr addrspace(1) %gates, i64 %decay.index
%decay.input.model = load i8, ptr addrspace(1) %decay.pointer, align 1
%decay.input = call float @recipe.decode(i8 %decay.input.model)
%softplus = call float @delta_softplus(float %decay.input)
%exponent = call float @recipe.state.mul(float %softplus, float %decay.scale)
%negated = call float @recipe.state.neg(float %exponent)
%decay = call float @recipe.state.exp(float %negated)
%write.index = add i64 %b.base, %time.wide
%write.pointer = getelementptr inbounds i8, ptr addrspace(1) %gates, i64 %write.index
%write.input.model = load i8, ptr addrspace(1) %write.pointer, align 1
%write.input = call float @recipe.decode(i8 %write.input.model)
%write = call float @recipe.state.sigmoid(float %write.input)
%weight.base = add i64 %vector.base, %vwidth.wide
br label %seed.row
seed.row: %seed.i = phi i32 [ 0, %entry ], [ %seed.i.next, %seed.row.done ] %seed.i.more = icmp ult i32 %seed.i, %kwidth
%seed.i.wide = zext i32 %seed.i to i64 %seed.i.row = mul i64 %seed.i.wide, %length.wide %seed.i.offset = add i64 %seed.i.row, %time.wide %seed.query.index = add i64 %q.base, %seed.i.offset
br i1 %seed.i.more, label %seed.column, label %column.loop
seed.column: %seed.j = phi i32 [ 0, %seed.row ], [ %seed.j.next, %seed.step ] %seed.j.more = icmp ult i32 %seed.j, %vwidth
br i1 %seed.j.more, label %seed.step, label %seed.row.done
seed.step: %seed.query.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %seed.query.index
%seed.query.model = load i8, ptr addrspace(1) %seed.query.pointer, align 1 %seed.query = call float @recipe.decode(i8 %seed.query.model)
%seed.j.wide = zext i32 %seed.j to i64 %seed.j.row = mul i64 %seed.j.wide, %length.wide %seed.j.offset = add i64 %seed.j.row, %time.wide %seed.delta.index = add i64 %o.base, %seed.j.offset
%seed.delta.pointer = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %seed.delta.index
%seed.delta = load float, ptr addrspace(1) %seed.delta.pointer, align 4
%seed.query.scaled = call float @recipe.state.mul(float %seed.query, float %query.scale)
%seed.product = call float @recipe.state.mul(float %seed.query.scaled, float %seed.delta)
%seed.row.base = mul i64 %seed.i.wide, %vwidth.wide %seed.cell = add i64 %seed.row.base, %seed.j.wide %seed.index = add i64 %adjoint.base, %seed.cell
%seed.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %seed.index
%seed.prior = load float, ptr addrspace(1) %seed.pointer, align 4
%seed.total = call float @recipe.state.add(float %seed.prior, float %seed.product)
store float %seed.total, ptr addrspace(1) %seed.pointer, align 4 %seed.j.next = add nuw i32 %seed.j, 1 br label %seed.column
seed.row.done: %seed.i.next = add nuw i32 %seed.i, 1 br label %seed.row
column.loop: %column = phi i32 [ 0, %seed.row ], [ %column.next, %column.done ]
%write.gradient = phi float [ %state.zero, %seed.row ], [ %write.gradient.next, %column.done ]
%column.wide = zext i32 %column to i64
%column.more = icmp ult i32 %column, %vwidth br i1 %column.more, label %column.row, label %row.loop
column.row: %column.i = phi i32 [ 0, %column.loop ], [ %column.i.next, %column.step ]
%readout = phi float [ %state.zero, %column.loop ], [ %readout.next, %column.step ]
%weight = phi float [ %state.zero, %column.loop ], [ %weight.next, %column.step ]
%column.i.more = icmp ult i32 %column.i, %kwidth br i1 %column.i.more, label %column.step, label %column.store
column.step: %column.i.wide = zext i32 %column.i to i64 %column.i.row = mul i64 %column.i.wide, %vwidth.wide %column.cell = add i64 %column.i.row, %column.wide
%column.state.index = add i64 %previous, %column.cell
%column.state.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %column.state.index
%column.state = load float, ptr addrspace(1) %column.state.pointer, align 4
%column.adjoint.index = add i64 %adjoint.base, %column.cell
%column.adjoint.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %column.adjoint.index
%column.adjoint = load float, ptr addrspace(1) %column.adjoint.pointer, align 4
%column.i.offset.row = mul i64 %column.i.wide, %length.wide %column.i.offset = add i64 %column.i.offset.row, %time.wide
%column.key.index = add i64 %k.base, %column.i.offset
%column.key.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %column.key.index
%column.key.model = load i8, ptr addrspace(1) %column.key.pointer, align 1 %column.key = call float @recipe.decode(i8 %column.key.model)
%column.decayed = call float @recipe.state.mul(float %decay, float %column.state)
%column.readout = call float @recipe.state.mul(float %column.key, float %column.decayed)
%readout.next = call float @recipe.state.add(float %readout, float %column.readout)
%column.weight = call float @recipe.state.mul(float %column.key, float %column.adjoint)
%weight.next = call float @recipe.state.add(float %weight, float %column.weight)
%column.i.next = add nuw i32 %column.i, 1 br label %column.row
column.store: %column.row.offset = mul i64 %column.wide, %length.wide %column.offset = add i64 %column.row.offset, %time.wide
%column.value.index = add i64 %v.base, %column.offset
%column.value.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %column.value.index
%column.value.model = load i8, ptr addrspace(1) %column.value.pointer, align 1 %column.value = call float @recipe.decode(i8 %column.value.model)
%readout.decayed = call float @recipe.state.mul(float %decay, float %readout)
%error = call float @recipe.state.sub(float %column.value, float %readout.decayed)
%error.index = add i64 %vector.base, %column.wide
%error.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %error.index
store float %error, ptr addrspace(1) %error.pointer, align 4
%weight.index = add i64 %weight.base, %column.wide
%weight.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %weight.index
store float %weight, ptr addrspace(1) %weight.pointer, align 4
%value.gradient = call float @recipe.state.mul(float %write, float %weight)
%value.adjoint.pointer = getelementptr inbounds float, ptr addrspace(1) %input.adjoint, i64 %column.value.index
%value.prior = load float, ptr addrspace(1) %value.adjoint.pointer, align 4
%value.total = call float @recipe.state.add(float %value.prior, float %value.gradient)
store float %value.total, ptr addrspace(1) %value.adjoint.pointer, align 4
%write.term = call float @recipe.state.mul(float %error, float %weight)
%write.gradient.next = call float @recipe.state.add(float %write.gradient, float %write.term) br label %column.done
column.done: %column.next = add nuw i32 %column, 1 br label %column.loop
row.loop: %row.i = phi i32 [ 0, %column.loop ], [ %row.i.next, %row.done ]
%decay.gradient = phi float [ %state.zero, %column.loop ], [ %decay.gradient.next, %row.done ]
%row.i.more = icmp ult i32 %row.i, %kwidth
%row.i.wide = zext i32 %row.i to i64 %row.i.offset.row = mul i64 %row.i.wide, %length.wide %row.i.offset = add i64 %row.i.offset.row, %time.wide
%row.key.index = add i64 %k.base, %row.i.offset %row.query.index = add i64 %q.base, %row.i.offset %row.i.base = mul i64 %row.i.wide, %vwidth.wide
br i1 %row.i.more, label %row.column, label %gates.entry
row.column: %row.j = phi i32 [ 0, %row.loop ], [ %row.j.next, %row.step ]
%decay.part = phi float [ %state.zero, %row.loop ], [ %decay.part.next, %row.step ]
%key.direct = phi float [ %state.zero, %row.loop ], [ %key.direct.next, %row.step ]
%key.readout = phi float [ %state.zero, %row.loop ], [ %key.readout.next, %row.step ]
%query.part = phi float [ %state.zero, %row.loop ], [ %query.part.next, %row.step ]
%row.j.more = icmp ult i32 %row.j, %vwidth br i1 %row.j.more, label %row.step, label %row.store
row.step: %row.j.wide = zext i32 %row.j to i64 %row.cell = add i64 %row.i.base, %row.j.wide %row.state.index = add i64 %previous, %row.cell
%row.state.pointer = getelementptr inbounds float, ptr addrspace(1) %context, i64 %row.state.index
%row.state = load float, ptr addrspace(1) %row.state.pointer, align 4
%row.adjoint.index = add i64 %adjoint.base, %row.cell
%row.adjoint.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %row.adjoint.index
%row.adjoint = load float, ptr addrspace(1) %row.adjoint.pointer, align 4
%row.error.index = add i64 %vector.base, %row.j.wide
%row.error.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %row.error.index
%row.error = load float, ptr addrspace(1) %row.error.pointer, align 4
%row.weight.index = add i64 %weight.base, %row.j.wide
%row.weight.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %row.weight.index
%row.weight = load float, ptr addrspace(1) %row.weight.pointer, align 4
%row.key.pointer = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %row.key.index
%row.key.model = load i8, ptr addrspace(1) %row.key.pointer, align 1 %row.key = call float @recipe.decode(i8 %row.key.model)
%row.direct.term = call float @recipe.state.mul(float %row.error, float %row.adjoint)
%key.direct.next = call float @recipe.state.add(float %key.direct, float %row.direct.term)
%row.state.decayed = call float @recipe.state.mul(float %decay, float %row.state)
%row.readout.term = call float @recipe.state.mul(float %row.weight, float %row.state.decayed)
%key.readout.next = call float @recipe.state.add(float %key.readout, float %row.readout.term)
%row.write.error = call float @recipe.state.mul(float %write, float %row.error)
%row.state.written = call float @recipe.state.mul(float %row.key, float %row.write.error)
%row.state.next = call float @recipe.state.add(float %row.state.decayed, float %row.state.written)
%row.j.offset.row = mul i64 %row.j.wide, %length.wide %row.j.offset = add i64 %row.j.offset.row, %time.wide
%row.delta.index = add i64 %o.base, %row.j.offset
%row.delta.pointer = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %row.delta.index
%row.delta = load float, ptr addrspace(1) %row.delta.pointer, align 4
%row.query.term = call float @recipe.state.mul(float %row.delta, float %row.state.next)
%query.part.next = call float @recipe.state.add(float %query.part, float %row.query.term)
%row.write.weight = call float @recipe.state.mul(float %write, float %row.weight)
%row.adjoint.written = call float @recipe.state.mul(float %row.key, float %row.write.weight)
%row.decayed.adjoint = call float @recipe.state.sub(float %row.adjoint, float %row.adjoint.written)
%row.adjoint.next = call float @recipe.state.mul(float %decay, float %row.decayed.adjoint)
%row.decay.term = call float @recipe.state.mul(float %row.state, float %row.decayed.adjoint)
%decay.part.next = call float @recipe.state.add(float %decay.part, float %row.decay.term)
store float %row.adjoint.next, ptr addrspace(1) %row.adjoint.pointer, align 4
%row.j.next = add nuw i32 %row.j, 1 br label %row.column
row.store: %key.readout.decayed = call float @recipe.state.mul(float %decay, float %key.readout)
%key.difference = call float @recipe.state.sub(float %key.direct, float %key.readout.decayed)
%key.gradient = call float @recipe.state.mul(float %write, float %key.difference)
%key.adjoint.pointer = getelementptr inbounds float, ptr addrspace(1) %input.adjoint, i64 %row.key.index
%key.prior = load float, ptr addrspace(1) %key.adjoint.pointer, align 4
%key.total = call float @recipe.state.add(float %key.prior, float %key.gradient)
store float %key.total, ptr addrspace(1) %key.adjoint.pointer, align 4
%query.adjoint.pointer = getelementptr inbounds float, ptr addrspace(1) %input.adjoint, i64 %row.query.index
%query.prior = load float, ptr addrspace(1) %query.adjoint.pointer, align 4
%query.gradient = call float @recipe.state.mul(float %query.part, float %query.scale)
%query.total = call float @recipe.state.add(float %query.prior, float %query.gradient)
store float %query.total, ptr addrspace(1) %query.adjoint.pointer, align 4
%decay.gradient.next = call float @recipe.state.add(float %decay.gradient, float %decay.part) br label %row.done
row.done: %row.i.next = add nuw i32 %row.i, 1 br label %row.loop
gates.entry: %decay.slope = call float @recipe.state.sigmoid(float %decay.input) %decay.factor = call float @recipe.state.mul(float %decay.scale, float %decay)
%decay.chain = call float @recipe.state.mul(float %decay.gradient, float %decay.factor)
%decay.chain.negated = call float @recipe.state.neg(float %decay.chain)
%decay.input.gradient = call float @recipe.state.mul(float %decay.chain.negated, float %decay.slope)
%decay.adjoint.pointer = getelementptr inbounds float, ptr addrspace(1) %gate.adjoint, i64 %decay.index
%decay.prior = load float, ptr addrspace(1) %decay.adjoint.pointer, align 4
%decay.total = call float @recipe.state.add(float %decay.prior, float %decay.input.gradient)
store float %decay.total, ptr addrspace(1) %decay.adjoint.pointer, align 4
%write.complement = call float @recipe.state.sub(float %state.one, float %write) %write.slope = call float @recipe.state.mul(float %write, float %write.complement)
%write.input.gradient = call float @recipe.state.mul(float %write.gradient, float %write.slope)
%write.adjoint.pointer = getelementptr inbounds float, ptr addrspace(1) %gate.adjoint, i64 %write.index
%write.prior = load float, ptr addrspace(1) %write.adjoint.pointer, align 4
%write.total = call float @recipe.state.add(float %write.prior, float %write.input.gradient)
store float %write.total, ptr addrspace(1) %write.adjoint.pointer, align 4
%scale.chain = call float @recipe.state.mul(float %softplus, float %decay.factor) %scale.negated = call float @recipe.state.neg(float %scale.chain)
%scale.gradient = call float @recipe.state.mul(float %decay.gradient, float %scale.negated) ret float %scale.gradient }
; One row and head of the gated delta rule in reverse. Each chunk is replayed
; forward from its committed entry state so the state before every position is
; available, then the chunk is walked backward. The decay scale gradient lands
; in this pair's partial for the fold below.
define internal void @delta_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %weights, ptr addrspace(1) %context, ptr addrspace(1) %backward,
ptr addrspace(1) %delta, ptr addrspace(1) %input.adjoint, ptr addrspace(1) %gate.adjoint,
i64 %p, i32 %kheads, i32 %kwidth, i32 %vheads, i32 %vwidth, i32 %length, i32 %chunk, i32 %chunks, i32 %pairs, i32 %tiled ) #3 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
%kheads.wide = zext i32 %kheads to i64 %kwidth.wide = zext i32 %kwidth to i64 %vheads.wide = zext i32 %vheads to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64 %chunk.wide = zext i32 %chunk to i64 %chunks.wide = zext i32 %chunks to i64 %pairs.wide = zext i32 %pairs to i64
%row = udiv i64 %p, %kheads.wide %khead = urem i64 %p, %kheads.wide %state = mul i64 %kwidth.wide, %vwidth.wide %state.i32 = trunc i64 %state to i32
%kchannels = mul i64 %kheads.wide, %kwidth.wide %kstream = mul i64 %kchannels, %length.wide
%vchannels = mul i64 %vheads.wide, %vwidth.wide %stream = mul i64 %vchannels, %length.wide
%kplanes = mul i64 %kstream, 2 %row.stride = add i64 %kplanes, %stream
%input.row = mul i64 %row, %row.stride %group = udiv i32 %vheads, %kheads %group.wide = zext i32 %group to i64
%khead.base = mul i64 %khead, %kwidth.wide %khead.offset = mul i64 %khead.base, %length.wide
%q.base = add i64 %input.row, %khead.offset %k.base = add i64 %q.base, %kstream
%value.plane = add i64 %input.row, %kplanes
%output.row = mul i64 %row, %stream
%gate.stream = mul i64 %vheads.wide, %length.wide %gate.row = mul i64 %row, %gate.stream %gate.pair = mul i64 %gate.row, 2
%entry.span = mul i64 %chunks.wide, %state
%work.region = mul i64 %pairs.wide, %entry.span
%pair.states = mul i64 %pairs.wide, %state %replay.region = add i64 %work.region, %pair.states
%replay.span = mul i64 %chunk.wide, %state
%replay.total = mul i64 %pairs.wide, %replay.span %adjoint.region = add i64 0, 0
%vector.region = add i64 %adjoint.region, %pair.states
%vector.span = mul i64 %vwidth.wide, 2
%vector.total = mul i64 %pairs.wide, %vector.span %partial.region = add i64 %vector.region, %vector.total
%pair.row = mul i64 %row, %vheads.wide
br label %head.loop
; Value heads cycle across key heads. Each sharing head walks in turn, so one thread owns the
; query and key adjoint elements of the head they share. One value head per key
; head makes exactly one pass and keeps the ungrouped indexing.
head.loop: %g = phi i32 [ 0, %entry ], [ %g.next, %head.done ] %g.more = icmp ult i32 %g, %group
br i1 %g.more, label %head.body, label %exit
head.body: %g.wide = zext i32 %g to i64 %head.grouped = add i64 %khead.first, %g.wide
%head.tile = mul i64 %g.wide, %kheads.wide %head.tiled = add i64 %head.tile, %khead %is.tiled = icmp ne i32 %tiled, 0
%head = select i1 %is.tiled, i64 %head.tiled, i64 %head.grouped %pair = add i64 %pair.row, %head
%head.base = mul i64 %head, %vwidth.wide %head.offset = mul i64 %head.base, %length.wide
%v.base = add i64 %value.plane, %head.offset %o.base = add i64 %output.row, %head.offset
%head.length = mul i64 %head, %length.wide %a.base = add i64 %gate.pair, %head.length %b.base = add i64 %a.base, %gate.stream
%entry.base = mul i64 %pair, %entry.span
%work.offset = mul i64 %pair, %state %work.base = add i64 %work.region, %work.offset
%replay.offset = mul i64 %pair, %replay.span %replay.base = add i64 %replay.region, %replay.offset
%adjoint.base = add i64 %adjoint.region, %work.offset
%vector.offset = mul i64 %pair, %vector.span %vector.base = add i64 %vector.region, %vector.offset
%partial.index = add i64 %partial.region, %pair
%head.i32 = trunc i64 %head to i32 %decay.pointer = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %head
%decay.parameter = load i8, ptr addrspace(1) %decay.pointer, align 1 %decay.scale = call i8 @recipe.exp(i8 %decay.parameter)
br label %zero.loop
zero.loop: %zero.i = phi i32 [ 0, %head.body ], [ %zero.next, %zero.step ] %zero.more = icmp ult i32 %zero.i, %state.i32
br i1 %zero.more, label %zero.step, label %chunk.loop
zero.step: %zero.i.wide = zext i32 %zero.i to i64 %zero.index = add i64 %adjoint.base, %zero.i.wide
%zero.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %zero.index
store float %state.zero, ptr addrspace(1) %zero.pointer, align 4 %zero.next = add nuw i32 %zero.i, 1 br label %zero.loop
chunk.loop: %remaining = phi i32 [ %chunks, %zero.loop ], [ %remaining.next, %chunk.done ]
%total = phi float [ %state.zero, %zero.loop ], [ %decay.sum, %chunk.done ]
%chunk.more = icmp ugt i32 %remaining, 0 %chunk.index = sub i32 %remaining, 1
%chunk.start = mul i32 %chunk.index, %chunk %chunk.index.wide = zext i32 %chunk.index to i64 %chunk.entry = mul i64 %chunk.index.wide, %state %chunk.entry.base = add i64 %entry.base, %chunk.entry
br i1 %chunk.more, label %restore.loop, label %store
restore.loop: %restore.i = phi i32 [ 0, %chunk.loop ], [ %restore.next, %restore.step ] %restore.more = icmp ult i32 %restore.i, %state.i32
br i1 %restore.more, label %restore.step, label %replay.loop
restore.step: %restore.i.wide = zext i32 %restore.i to i64 %restore.entry = add i64 %chunk.entry.base, %restore.i.wide
%restore.entry.pointer = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %restore.entry
%restore.value = load i8, ptr addrspace(1) %restore.entry.pointer, align 1 %restore.work = add i64 %work.base, %restore.i.wide
%restore.work.pointer = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %restore.work
store i8 %restore.value, ptr addrspace(1) %restore.work.pointer, align 1 %restore.next = add nuw i32 %restore.i, 1 br label %restore.loop
replay.loop: %replay.i = phi i32 [ 0, %restore.loop ], [ %replay.i.next, %replay.step ] %replay.time = add i32 %chunk.start, %replay.i
%replay.i.more = icmp ult i32 %replay.i, %chunk %replay.time.more = icmp ult i32 %replay.time, %length
%replay.more = and i1 %replay.i.more, %replay.time.more
%replay.i.wide = zext i32 %replay.i to i64 %replay.slot.offset = mul i64 %replay.i.wide, %state %replay.slot = add i64 %replay.base, %replay.slot.offset
br i1 %replay.more, label %replay.save, label %backward.loop
replay.save: %save.i = phi i32 [ 0, %replay.loop ], [ %save.next, %save.step ] %save.more = icmp ult i32 %save.i, %state.i32
br i1 %save.more, label %save.step, label %replay.step
save.step: %save.i.wide = zext i32 %save.i to i64 %save.work = add i64 %work.base, %save.i.wide
%save.work.pointer = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %save.work
%save.value = load i8, ptr addrspace(1) %save.work.pointer, align 1 %save.slot = add i64 %replay.slot, %save.i.wide
%save.slot.pointer = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %save.slot
store i8 %save.value, ptr addrspace(1) %save.slot.pointer, align 1 %save.next = add nuw i32 %save.i, 1 br label %replay.save
replay.step: call void @delta_step( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %context, ptr addrspace(1) %context,
i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base, i64 %work.base,
i32 %kwidth, i32 %vwidth, i32 %length, i32 %replay.time, i8 %decay.scale, i1 false, i32 0, i32 1, ptr addrspace(1) null )
%replay.i.next = add nuw i32 %replay.i, 1 br label %replay.loop
backward.loop: %backward.i = phi i32 [ %replay.i, %replay.loop ], [ %backward.index, %backward.step ]
%decay.sum = phi float [ %total, %replay.loop ], [ %decay.sum.next, %backward.step ]
%backward.more = icmp ugt i32 %backward.i, 0 br i1 %backward.more, label %backward.step, label %chunk.done
backward.step: %backward.index = sub i32 %backward.i, 1 %backward.time = add i32 %chunk.start, %backward.index %backward.index.wide = zext i32 %backward.index to i64
%backward.slot.offset = mul i64 %backward.index.wide, %state %backward.slot = add i64 %replay.base, %backward.slot.offset
%contribution = call float @delta_back( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %context, ptr addrspace(1) %backward, ptr addrspace(1) %delta,
ptr addrspace(1) %input.adjoint, ptr addrspace(1) %gate.adjoint, i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base,
i64 %backward.slot, i64 %adjoint.base, i64 %vector.base, i32 %kwidth, i32 %vwidth, i32 %length, i32 %backward.time, i8 %decay.scale )
%decay.sum.next = call float @recipe.state.add(float %decay.sum, float %contribution) br label %backward.loop
chunk.done: %remaining.next = sub i32 %remaining, 1 br label %chunk.loop
store: %partial.pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %partial.index
store float %total, ptr addrspace(1) %partial.pointer, align 4 br label %head.done
head.done: %g.next = add nuw i32 %g, 1 br label %head.loop
exit: ret void }
; Decay scale %p sums its per-row partials in row order and writes the gradient.
define internal void @delta_reverse_decay_body( ptr addrspace(1) %backward, ptr addrspace(1) %gradient, i64 %p, i32 %rows, i32 %heads, i32 %partials, i32 %offset ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
%heads.wide = zext i32 %heads to i64 %partials.wide = zext i32 %partials to i64 %offset.wide = zext i32 %offset to i64
br label %loop loop: %r = phi i32 [ 0, %entry ], [ %next, %step ] %sum = phi float [ %state.zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i32 %r, %rows br i1 %more, label %step, label %done
step: %r.wide = zext i32 %r to i64 %pair.base = mul i64 %r.wide, %heads.wide %pair = add i64 %pair.base, %p %index = add i64 %partials.wide, %pair
%pointer = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %index
%value = load float, ptr addrspace(1) %pointer, align 4 %sum.next = call float @recipe.state.add(float %sum, float %value)
%next = add nuw i32 %r, 1 br label %loop
done: %gradient.index = add i64 %offset.wide, %p
%gradient.pointer = getelementptr inbounds float, ptr addrspace(1) %gradient, i64 %gradient.index
store float %sum, ptr addrspace(1) %gradient.pointer, align 4 ret void }
; Mixture routing bodies. Scores and weights hold %experts values per position,
; expert e of position t at [e * length + t] within the row.
; Position %p keeps its %top highest scores, ties to the lower expert, under a
; softmax over the kept scores, and zero elsewhere. The weights buffer holds
; the selection marks until the last pass overwrites them.
; One expert unnormalized routing score: exp(score - maximum) under a softmax,
; the logistic of the score under a sigmoid.
define internal i8 @topk_score( i8 %score, i8 %maximum, i1 %sigmoid ) #1 { entry:
%shifted = call i8 @recipe.sub(i8 %score, i8 %maximum) %exponential = call i8 @recipe.exp(i8 %shifted)
%logistic = call i8 @sigmoid(i8 %score) %result = select i1 %sigmoid, i8 %logistic, i8 %exponential ret i8 %result }
; The routing weights of one position. The top scores are kept by rank, scored
; by softmax over every expert or by sigmoid, and divided by the kept total when
; the block renormalizes. A plain softmax divides by every expert instead, which
; is the evaluate-all-then-mask reference; a plain sigmoid divides by nothing.
define internal void @topk_forward_body( ptr addrspace(1) %scores, ptr addrspace(1) %weights, ptr addrspace(1) %selected, i64 %p, i32 %experts, i32 %length, i32 %top, i32 %scoring, i32 %renormalize ) #1 { entry:
%experts.wide = zext i32 %experts to i64 %length.wide = zext i32 %length to i64 %top.wide = zext i32 %top to i64
%selected.stride = add i64 %top.wide, 1 %selected.base = mul i64 %p, %selected.stride
%row = udiv i64 %p, %length.wide %position = urem i64 %p, %length.wide %per.row = mul i64 %experts.wide, %length.wide %row.base = mul i64 %row, %per.row %base = add i64 %row.base, %position
%sigmoid = icmp ne i32 %scoring, 0 %renorm = icmp ne i32 %renormalize, 0 %every = xor i1 %renorm, true %plain = xor i1 %sigmoid, true %divide = or i1 %renorm, %plain
br label %clear.loop clear.loop: %clear = phi i64 [ 0, %entry ], [ %clear.next, %clear.step ] %clear.nan = phi i1 [ false, %entry ], [ %clear.nan.next, %clear.step ] %clear.more = icmp ult i64 %clear, %experts.wide
br i1 %clear.more, label %clear.step, label %select.entry clear.step: %clear.offset = mul i64 %clear, %length.wide %clear.index = add i64 %base, %clear.offset
%clear.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %clear.index store i8 0, ptr addrspace(1) %clear.ptr, align 1
%clear.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %clear.index %clear.score = load i8, ptr addrspace(1) %clear.score.ptr, align 1
%clear.ordered = call i1 @recipe.oeq(i8 %clear.score, i8 %clear.score) %clear.unordered = xor i1 %clear.ordered, true %clear.nan.next = or i1 %clear.nan, %clear.unordered
%clear.next = add i64 %clear, 1 br label %clear.loop
; Keep the best candidates in the existing selected-index storage. Most
; candidates only compare against the current cutoff. Equal scores retain the
; earlier expert. Unordered scores keep the original scan-order behavior.
select.entry: br i1 %clear.nan, label %select.loop, label %rank.loop
rank.loop:
%rank.expert = phi i64 [ 0, %select.entry ], [ %rank.next, %rank.advance ]
%rank.count = phi i64 [ 0, %select.entry ], [ %rank.count.next, %rank.advance ]
%rank.cutoff.value = phi i8 [ 0, %select.entry ], [ %rank.cutoff.next, %rank.advance ]
%rank.more = icmp ult i64 %rank.expert, %experts.wide
br i1 %rank.more, label %rank.score, label %rank.mark.loop
rank.score:
%rank.offset = mul i64 %rank.expert, %length.wide %rank.index = add i64 %base, %rank.offset
%rank.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %rank.index %rank.score.value = load i8, ptr addrspace(1) %rank.score.ptr, align 1
%rank.full = icmp eq i64 %rank.count, %top.wide
%rank.last = sub i64 %top.wide, 1 %rank.start = select i1 %rank.full, i64 %rank.last, i64 %rank.count
br i1 %rank.full, label %rank.cutoff, label %rank.insert
rank.cutoff:
%rank.admitted = call i1 @recipe.ogt(i8 %rank.score.value, i8 %rank.cutoff.value)
br i1 %rank.admitted, label %rank.insert, label %rank.advance
rank.insert: br label %rank.shift.loop
rank.shift.loop:
%rank.at = phi i64 [ %rank.start, %rank.insert ], [ %rank.prior, %rank.shift ]
%rank.has.prior = icmp ugt i64 %rank.at, 0 br i1 %rank.has.prior, label %rank.compare, label %rank.store
rank.compare:
%rank.prior = sub i64 %rank.at, 1 %rank.prior.index = add i64 %selected.base, %rank.at
%rank.prior.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %rank.prior.index %rank.prior.id = load i32, ptr addrspace(1) %rank.prior.ptr, align 4
%rank.prior.wide = zext i32 %rank.prior.id to i64 %rank.prior.offset = mul i64 %rank.prior.wide, %length.wide %rank.prior.score.index = add i64 %base, %rank.prior.offset
%rank.prior.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %rank.prior.score.index %rank.prior.score = load i8, ptr addrspace(1) %rank.prior.score.ptr, align 1
%rank.higher = call i1 @recipe.ogt(i8 %rank.score.value, i8 %rank.prior.score) br i1 %rank.higher, label %rank.shift, label %rank.store
rank.shift:
%rank.shift.local = add i64 %rank.at, 1 %rank.shift.index = add i64 %selected.base, %rank.shift.local %rank.shift.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %rank.shift.index
store i32 %rank.prior.id, ptr addrspace(1) %rank.shift.ptr, align 4 br label %rank.shift.loop
rank.store:
%rank.store.local = add i64 %rank.at, 1 %rank.store.index = add i64 %selected.base, %rank.store.local %rank.store.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %rank.store.index
%rank.id = trunc i64 %rank.expert to i32 store i32 %rank.id, ptr addrspace(1) %rank.store.ptr, align 4
%rank.grown = add i64 %rank.count, 1 %rank.stored.count = select i1 %rank.full, i64 %rank.count, i64 %rank.grown
; The cutoff changes only when a candidate enters the selected list.
%rank.cutoff.index = add i64 %selected.base, %rank.stored.count %rank.cutoff.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %rank.cutoff.index
%rank.cutoff.id = load i32, ptr addrspace(1) %rank.cutoff.ptr, align 4 %rank.cutoff.wide = zext i32 %rank.cutoff.id to i64
%rank.cutoff.offset = mul i64 %rank.cutoff.wide, %length.wide %rank.cutoff.score.index = add i64 %base, %rank.cutoff.offset
%rank.cutoff.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %rank.cutoff.score.index %rank.cutoff.score = load i8, ptr addrspace(1) %rank.cutoff.score.ptr, align 1
br label %rank.advance
rank.advance:
%rank.count.next = phi i64 [ %rank.count, %rank.cutoff ], [ %rank.stored.count, %rank.store ]
%rank.cutoff.next = phi i8 [ %rank.cutoff.value, %rank.cutoff ], [ %rank.cutoff.score, %rank.store ]
%rank.next = add i64 %rank.expert, 1 br label %rank.loop
rank.mark.loop:
%rank.mark = phi i64 [ 0, %rank.loop ], [ %rank.mark.next, %rank.mark.step ] %rank.mark.more = icmp ult i64 %rank.mark, %rank.count
br i1 %rank.mark.more, label %rank.mark.step, label %normalize.entry
rank.mark.step:
%rank.mark.local = add i64 %rank.mark, 1 %rank.mark.index = add i64 %selected.base, %rank.mark.local %rank.mark.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %rank.mark.index
%rank.mark.id = load i32, ptr addrspace(1) %rank.mark.ptr, align 4 %rank.mark.wide = zext i32 %rank.mark.id to i64
%rank.mark.offset = mul i64 %rank.mark.wide, %length.wide %rank.mark.weight.index = add i64 %base, %rank.mark.offset %rank.mark.weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %rank.mark.weight.index
store i8 56, ptr addrspace(1) %rank.mark.weight.ptr, align 1 %rank.mark.next = add i64 %rank.mark, 1 br label %rank.mark.loop
select.loop: %pick = phi i64 [ 0, %select.entry ], [ %pick.next, %select.mark ] %pick.more = icmp ult i64 %pick, %top.wide br i1 %pick.more, label %scan.entry, label %normalize.entry
scan.entry: br label %scan.loop
scan.loop: %candidate = phi i64 [ 0, %scan.entry ], [ %candidate.next, %scan.step ] %best = phi i64 [ -1, %scan.entry ], [ %best.next, %scan.step ]
%best.score = phi i8 [ 0, %scan.entry ], [ %best.score.next, %scan.step ] %scan.more = icmp ult i64 %candidate, %experts.wide br i1 %scan.more, label %scan.step, label %select.mark
scan.step: %candidate.offset = mul i64 %candidate, %length.wide %candidate.index = add i64 %base, %candidate.offset
%mark.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %candidate.index %mark = load i8, ptr addrspace(1) %mark.ptr, align 1 %unmarked = call i1 @recipe.oeq(i8 %mark, i8 0)
%score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %candidate.index %score = load i8, ptr addrspace(1) %score.ptr, align 1
%none = icmp eq i64 %best, -1 %higher = call i1 @recipe.ogt(i8 %score, i8 %best.score) %better = or i1 %none, %higher %take = and i1 %unmarked, %better
%best.next = select i1 %take, i64 %candidate, i64 %best %best.score.next = select i1 %take, i8 %score, i8 %best.score %candidate.next = add i64 %candidate, 1 br label %scan.loop
select.mark: %best.offset = mul i64 %best, %length.wide %best.index = add i64 %base, %best.offset %best.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %best.index
store i8 56, ptr addrspace(1) %best.ptr, align 1 %pick.next = add i64 %pick, 1 br label %select.loop
normalize.entry: br label %max.loop
max.loop: %m = phi i64 [ 0, %normalize.entry ], [ %m.next, %max.step ] %maximum = phi i8 [ 0, %normalize.entry ], [ %maximum.next, %max.step ] %m.first = phi i1 [ true, %normalize.entry ], [ %m.first.next, %max.step ]
%max.more = icmp ult i64 %m, %experts.wide br i1 %max.more, label %max.step, label %sum.entry
max.step: %m.offset = mul i64 %m, %length.wide %m.index = add i64 %base, %m.offset %m.mark.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %m.index %m.mark = load i8, ptr addrspace(1) %m.mark.ptr, align 1
%m.marked = call i1 @recipe.oeq(i8 %m.mark, i8 56) %m.member = or i1 %m.marked, %every
%m.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %m.index %m.score = load i8, ptr addrspace(1) %m.score.ptr, align 1
%m.higher = call i1 @recipe.ogt(i8 %m.score, i8 %maximum) %m.better = or i1 %m.first, %m.higher %m.take = and i1 %m.member, %m.better
%maximum.next = select i1 %m.take, i8 %m.score, i8 %maximum %m.first.next = select i1 %m.take, i1 false, i1 %m.first %m.next = add i64 %m, 1 br label %max.loop
sum.entry: br label %sum.loop
sum.loop: %s = phi i64 [ 0, %sum.entry ], [ %s.next, %sum.store ] %total = phi i8 [ 0, %sum.entry ], [ %total.next, %sum.store ] %sum.more = icmp ult i64 %s, %experts.wide br i1 %sum.more, label %sum.step, label %write.entry
sum.step: %s.offset = mul i64 %s, %length.wide %s.index = add i64 %base, %s.offset %s.mark.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %s.index %s.mark = load i8, ptr addrspace(1) %s.mark.ptr, align 1
%s.marked = call i1 @recipe.oeq(i8 %s.mark, i8 56) %s.member = or i1 %s.marked, %every
br i1 %s.member, label %sum.score, label %sum.store
sum.score:
%s.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %s.index %s.score = load i8, ptr addrspace(1) %s.score.ptr, align 1
%s.raw = call i8 @topk_score(i8 %s.score, i8 %maximum, i1 %sigmoid) br label %sum.store
sum.store: %s.term = phi i8 [ %s.raw, %sum.score ], [ 0, %sum.step ]
%s.kept = select i1 %s.marked, i8 %s.term, i8 0 store i8 %s.kept, ptr addrspace(1) %s.mark.ptr, align 1
%total.next = call i8 @recipe.add(i8 %total, i8 %s.term) %s.next = add i64 %s, 1 br label %sum.loop
write.entry: %denominator = select i1 %divide, i8 %total, i8 56 br label %write.loop
write.loop: %w = phi i64 [ 0, %write.entry ], [ %w.next, %write.advance ] %kept = phi i64 [ 0, %write.entry ], [ %kept.next, %write.advance ] %write.more = icmp ult i64 %w, %experts.wide br i1 %write.more, label %write.step, label %exit
write.step: %w.offset = mul i64 %w, %length.wide %w.index = add i64 %base, %w.offset %w.mark.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %w.index %w.mark = load i8, ptr addrspace(1) %w.mark.ptr, align 1
%w.probability = call i8 @recipe.div(i8 %w.mark, i8 %denominator)
%w.unselected = call i1 @recipe.oeq(i8 %w.mark, i8 0)
%w.value = select i1 %w.unselected, i8 0, i8 %w.probability
store i8 %w.value, ptr addrspace(1) %w.mark.ptr, align 1
%w.zero = call i1 @recipe.oeq(i8 %w.value, i8 0) br i1 %w.zero, label %write.advance, label %write.selected
write.selected: %selected.local = add i64 %kept, 1 %selected.index = add i64 %selected.base, %selected.local
%selected.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %selected.index %selected.expert = trunc i64 %w to i32
store i32 %selected.expert, ptr addrspace(1) %selected.ptr, align 4 %kept.grown = add i64 %kept, 1 br label %write.advance
write.advance: %kept.next = phi i64 [ %kept, %write.step ], [ %kept.grown, %write.selected ] %w.next = add i64 %w, 1 br label %write.loop
exit: %selected.count = trunc i64 %kept to i32 %selected.count.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %selected.base
store i32 %selected.count, ptr addrspace(1) %selected.count.ptr, align 4 ret void }
; The routing adjoint of one position. Each score receives its own slope over
; the divisor times its delta less the kept mean; renormalizing confines that to
; the kept experts, while a plain softmax also reaches the experts it dropped.
define internal void @topk_reverse_body( ptr addrspace(1) %scores, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %experts, i32 %length, i32 %scoring, i32 %renormalize ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%experts.wide = zext i32 %experts to i64 %length.wide = zext i32 %length to i64
%row = udiv i64 %p, %length.wide %position = urem i64 %p, %length.wide %per.row = mul i64 %experts.wide, %length.wide %row.base = mul i64 %row, %per.row %base = add i64 %row.base, %position
%sigmoid = icmp ne i32 %scoring, 0 %renorm = icmp ne i32 %renormalize, 0 %every = xor i1 %renorm, true %plain = xor i1 %sigmoid, true %divide = or i1 %renorm, %plain
br label %max.loop
max.loop: %m = phi i64 [ 0, %entry ], [ %m.next, %max.step ] %maximum = phi i8 [ 0, %entry ], [ %maximum.next, %max.step ] %m.first = phi i1 [ true, %entry ], [ %m.first.next, %max.step ]
%max.more = icmp ult i64 %m, %experts.wide br i1 %max.more, label %max.step, label %sum.entry
max.step: %m.offset = mul i64 %m, %length.wide %m.index = add i64 %base, %m.offset
%m.weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %m.index %m.weight = load i8, ptr addrspace(1) %m.weight.ptr, align 1
%m.zero = call i1 @recipe.oeq(i8 %m.weight, i8 0) %m.marked = xor i1 %m.zero, true %m.member = or i1 %m.marked, %every
%m.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %m.index %m.score = load i8, ptr addrspace(1) %m.score.ptr, align 1
%m.higher = call i1 @recipe.ogt(i8 %m.score, i8 %maximum) %m.better = or i1 %m.first, %m.higher %m.take = and i1 %m.member, %m.better
%maximum.next = select i1 %m.take, i8 %m.score, i8 %maximum %m.first.next = select i1 %m.take, i1 false, i1 %m.first %m.next = add i64 %m, 1 br label %max.loop
sum.entry: br label %sum.loop
sum.loop: %s = phi i64 [ 0, %sum.entry ], [ %s.next, %sum.step ] %total = phi float [ %state.zero, %sum.entry ], [ %total.next, %sum.step ] %inner = phi float [ %state.zero, %sum.entry ], [ %inner.next, %sum.step ]
%sum.more = icmp ult i64 %s, %experts.wide br i1 %sum.more, label %sum.step, label %write.entry
sum.step: %s.offset = mul i64 %s, %length.wide %s.index = add i64 %base, %s.offset
%s.weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %s.index %s.weight.model = load i8, ptr addrspace(1) %s.weight.ptr, align 1 %s.weight = call float @recipe.decode(i8 %s.weight.model)
%s.zero = call i1 @recipe.oeq(i8 %s.weight.model, i8 0) %s.marked = xor i1 %s.zero, true %s.member = or i1 %s.marked, %every
%s.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %s.index %s.score = load i8, ptr addrspace(1) %s.score.ptr, align 1
%s.raw.model = call i8 @topk_score(i8 %s.score, i8 %maximum, i1 %sigmoid) %s.raw = call float @recipe.decode(i8 %s.raw.model) %s.term = select i1 %s.member, float %s.raw, float %state.zero
%total.next = call float @recipe.state.add(float %total, float %s.term)
%s.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %s.index %s.delta = load float, ptr addrspace(1) %s.delta.ptr, align 4
%s.product = call float @recipe.state.mul(float %s.weight, float %s.delta) %inner.next = call float @recipe.state.add(float %inner, float %s.product) %s.next = add i64 %s, 1 br label %sum.loop
write.entry: %denominator = select i1 %divide, float %total, float %state.one %subtract = select i1 %divide, float %inner, float %state.zero br label %write.loop
write.loop: %w = phi i64 [ 0, %write.entry ], [ %w.next, %write.step ] %write.more = icmp ult i64 %w, %experts.wide br i1 %write.more, label %write.step, label %exit
write.step: %w.offset = mul i64 %w, %length.wide %w.index = add i64 %base, %w.offset
%w.weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %w.index %w.weight.model = load i8, ptr addrspace(1) %w.weight.ptr, align 1 %w.weight = call float @recipe.decode(i8 %w.weight.model)
%w.zero = call i1 @recipe.oeq(i8 %w.weight.model, i8 0) %w.marked = xor i1 %w.zero, true
%w.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %w.index %w.delta = load float, ptr addrspace(1) %w.delta.ptr, align 4
%w.score.ptr = getelementptr inbounds i8, ptr addrspace(1) %scores, i64 %w.index %w.score = load i8, ptr addrspace(1) %w.score.ptr, align 1
%w.raw.model = call i8 @topk_score(i8 %w.score, i8 %maximum, i1 %sigmoid) %w.raw = call float @recipe.decode(i8 %w.raw.model)
%w.rest = call float @recipe.state.sub(float %state.one, float %w.raw) %w.logistic = call float @recipe.state.mul(float %w.raw, float %w.rest) %w.slope = select i1 %sigmoid, float %w.logistic, float %w.raw
%w.coefficient = call float @recipe.state.div(float %w.slope, float %denominator) %w.own = select i1 %w.marked, float %w.delta, float %state.zero
%w.centered = call float @recipe.state.sub(float %w.own, float %subtract) %w.term = call float @recipe.state.mul(float %w.coefficient, float %w.centered)
%w.blocked = and i1 %renorm, %w.zero %w.value = select i1 %w.blocked, float %state.zero, float %w.term
%w.adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %w.index %w.prior = load float, ptr addrspace(1) %w.adjoint.ptr, align 4
%w.sum = call float @recipe.state.add(float %w.prior, float %w.value) store float %w.sum, ptr addrspace(1) %w.adjoint.ptr, align 4 %w.next = add i64 %w, 1 br label %write.loop
exit: ret void }
; Mixture dispatch bodies. A position selects the experts whose routing weight
; is not zero; slot j is the j-th of those in ascending expert order. The gate
; and up tables hold [expert][hidden][channel], the down table holds
; [expert][channel][hidden].
; Expert %p lists the positions routed to it, in ascending position order, as
; (position, slot) pairs after the per-expert counts at the front of %context.
define internal void @moe_bucket_body( ptr addrspace(1) %routing, ptr addrspace(1) %context, i64 %p, i32 %pairs, i32 %length, i32 %experts, i32 %top ) #1 { entry:
%pairs.wide = zext i32 %pairs to i64 %length.wide = zext i32 %length to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%per.row = mul i64 %experts.wide, %length.wide %expert.base = mul i64 %p, %length.wide
br label %base.loop
base.loop: %b = phi i32 [ 0, %entry ], [ %b.next, %base.done ] %base = phi i64 [ 0, %entry ], [ %b.lower, %base.done ]
%base.more = icmp ult i32 %b, %pairs br i1 %base.more, label %base.step, label %fill.entry
base.step: %b.wide = zext i32 %b to i64 %b.row = udiv i64 %b.wide, %length.wide %b.position = urem i64 %b.wide, %length.wide %b.row.base = mul i64 %b.row, %per.row %b.pair = add i64 %b.row.base, %b.position
br label %base.lower
base.lower: %bc = phi i32 [ 0, %base.step ], [ %bc.next, %base.lower.step ] %b.lower = phi i64 [ %base, %base.step ], [ %b.lower.next, %base.lower.step ]
%bc.wide = zext i32 %bc to i64 %bc.more = icmp ult i64 %bc.wide, %p br i1 %bc.more, label %base.lower.step, label %base.done
base.lower.step: %bc.offset = mul i64 %bc.wide, %length.wide %bc.index = add i64 %b.pair, %bc.offset
%bc.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %bc.index %bc.weight = load i8, ptr addrspace(1) %bc.ptr, align 1
%bc.zero = call i1 @recipe.oeq(i8 %bc.weight, i8 0) %bc.taken = xor i1 %bc.zero, true %bc.step = zext i1 %bc.taken to i32
%bc.step.wide = zext i32 %bc.step to i64 %b.lower.next = add i64 %b.lower, %bc.step.wide %bc.next = add i32 %bc, 1 br label %base.lower
base.done: %b.next = add i32 %b, 1 br label %base.loop
fill.entry: %experts.base = zext i32 %experts to i64 %start = add i64 %experts.base, %base br label %fill.loop
fill.loop: %i = phi i32 [ 0, %fill.entry ], [ %i.next, %advance ] %cursor = phi i32 [ 0, %fill.entry ], [ %cursor.next, %advance ]
%more = icmp ult i32 %i, %pairs br i1 %more, label %step, label %done
step: %i.wide = zext i32 %i to i64 %row = udiv i64 %i.wide, %length.wide %position = urem i64 %i.wide, %length.wide %row.base = mul i64 %row, %per.row %pair.base = add i64 %row.base, %position
%own.index = add i64 %pair.base, %expert.base %own.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %own.index
%own.weight = load i8, ptr addrspace(1) %own.ptr, align 1 %own.zero = call i1 @recipe.oeq(i8 %own.weight, i8 0) %own.taken = xor i1 %own.zero, true
br i1 %own.taken, label %slot.entry, label %advance
slot.entry: br label %slot.loop
slot.loop: %c = phi i32 [ 0, %slot.entry ], [ %c.next, %slot.step ] %slot = phi i32 [ 0, %slot.entry ], [ %slot.next, %slot.step ]
%c.wide = zext i32 %c to i64 %slot.more = icmp ult i64 %c.wide, %p br i1 %slot.more, label %slot.step, label %write
slot.step: %c.offset = mul i64 %c.wide, %length.wide %c.index = add i64 %pair.base, %c.offset
%c.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %c.index %c.weight = load i8, ptr addrspace(1) %c.ptr, align 1
%c.zero = call i1 @recipe.oeq(i8 %c.weight, i8 0) %c.taken = xor i1 %c.zero, true %c.step = zext i1 %c.taken to i32
%slot.next = add i32 %slot, %c.step %c.next = add i32 %c, 1 br label %slot.loop
write: %scaled = mul i32 %i, %top %entry.value = add i32 %scaled, %slot %cursor.wide = zext i32 %cursor to i64 %write.index = add i64 %start, %cursor.wide
%write.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %write.index store i32 %entry.value, ptr addrspace(1) %write.ptr, align 4
%cursor.grown = add i32 %cursor, 1 br label %advance
advance: %cursor.next = phi i32 [ %cursor.grown, %write ], [ %cursor, %step ] %i.next = add i32 %i, 1 br label %fill.loop
done: %count.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %p store i32 %cursor, ptr addrspace(1) %count.ptr, align 4 ret void }
; Output element %p of a gate or up projection reads one row of the slice
; belonging to the (slot + 1)-th selected expert of its position.
define internal float @expert_wave_sum(float %value, i32 %lane, i32 %width) #1 { entry:
%first = udiv i32 %width, 2 br label %reduce.loop
reduce.loop: %offset = phi i32 [ %first, %entry ], [ %next, %reduce.step ] %sum = phi float [ %value, %entry ], [ %combined, %reduce.step ]
%more = icmp ugt i32 %offset, 0 br i1 %more, label %reduce.step, label %done
reduce.step: %other = xor i32 %lane, %offset %index = mul i32 %other, 4
%partner = call float @recipe.wave.partner(float %sum, i32 %index)
%combined = call float @recipe.state.add(float %sum, float %partner) %next = udiv i32 %offset, 2 br label %reduce.loop
done: ret float %sum }
define internal void @expert_in_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %selected, ptr addrspace(1) %weights, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %decode, i32 %lane, i32 %width ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
%owner = icmp eq i32 %lane, 0
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%units = mul i64 %top.wide, %hidden.wide %narrow = mul i64 %units, %length.wide
%row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %unit = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%slot = udiv i64 %unit, %hidden.wide %f = urem i64 %unit, %hidden.wide %slot.i32 = trunc i64 %slot to i32
%selected.row = mul i64 %row, %length.wide %selected.position = add i64 %selected.row, %position %selected.stride = add i64 %top.wide, 1
%selected.base = mul i64 %selected.position, %selected.stride %selected.count.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %selected.base
%selected.count = load i32, ptr addrspace(1) %selected.count.ptr, align 4 %found = icmp ult i32 %slot.i32, %selected.count
br i1 %found, label %sum.entry, label %absent
sum.entry: %selected.local = add i64 %slot, 1 %selected.index = add i64 %selected.base, %selected.local
%selected.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %selected.index %expert = load i32, ptr addrspace(1) %selected.ptr, align 4
%plane = mul i64 %hidden.wide, %channels.wide %expert.wide = zext i32 %expert to i64 %slice = mul i64 %expert.wide, %plane %f.row = mul i64 %f, %channels.wide %weight.base = add i64 %slice, %f.row
%input.channels = mul i64 %channels.wide, %length.wide %input.row = mul i64 %row, %input.channels %input.base = add i64 %input.row, %position
br label %sum.loop
sum.loop: %k = phi i32 [ %lane, %sum.entry ], [ %k.next, %sum.step ] %total = phi float [ %state.zero, %sum.entry ], [ %total.next, %sum.step ]
%sum.more = icmp ult i32 %k, %channels br i1 %sum.more, label %sum.step, label %done
sum.step: %k.wide = zext i32 %k to i64 %weight.index = add i64 %weight.base, %k.wide
%weight = call i8 @recipe.model.weight(ptr addrspace(1) %weights, i64 %weight.index, i32 %decode)
%input.offset = mul i64 %k.wide, %length.wide %input.index = add i64 %input.base, %input.offset
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %input.index %value = load i8, ptr addrspace(1) %input.ptr, align 1
%weight.state = call float @recipe.decode(i8 %weight)
%value.state = call float @recipe.decode(i8 %value)
%product = call float @recipe.state.mul(float %weight.state, float %value.state)
%total.next = call float @recipe.state.add(float %total, float %product) %k.next = add i32 %k, %width br label %sum.loop
done: %sum = call float @expert_wave_sum(float %total, i32 %lane, i32 %width)
br i1 %owner, label %store, label %exit
store: %result = call i8 @recipe.encode(float %sum)
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %result, ptr addrspace(1) %output.ptr, align 1 br label %exit
absent: br i1 %owner, label %clear, label %exit
clear: %absent.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 0, ptr addrspace(1) %absent.ptr, align 1 br label %exit
exit: ret void }
; Input element %p of a gate or up projection collects every selected expert's
; column, in ascending expert order.
define internal void @expert_in_reverse_input_body( ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %channel = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
%units = mul i64 %top.wide, %hidden.wide %delta.narrow = mul i64 %units, %length.wide %delta.row = mul i64 %row, %delta.narrow %delta.base = add i64 %delta.row, %position
%plane = mul i64 %hidden.wide, %channels.wide
br label %expert.loop
expert.loop: %e = phi i32 [ 0, %entry ], [ %e.next, %advance ] %slot = phi i32 [ 0, %entry ], [ %slot.next, %advance ] %total = phi float [ %state.zero, %entry ], [ %total.next, %advance ]
%expert.more = icmp ult i32 %e, %experts br i1 %expert.more, label %step, label %done
step: %e.wide = zext i32 %e to i64 %e.offset = mul i64 %e.wide, %length.wide %e.index = add i64 %routing.base, %e.offset
%e.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %e.index %e.weight = load i8, ptr addrspace(1) %e.ptr, align 1
%e.zero = call i1 @recipe.oeq(i8 %e.weight, i8 0) %e.taken = xor i1 %e.zero, true
br i1 %e.taken, label %hidden.entry, label %advance
hidden.entry: %slice = mul i64 %e.wide, %plane %slot.wide = zext i32 %slot to i64 %slot.offset = mul i64 %slot.wide, %hidden.wide br label %hidden.loop
hidden.loop: %f = phi i32 [ 0, %hidden.entry ], [ %f.next, %hidden.step ] %inner = phi float [ %state.zero, %hidden.entry ], [ %inner.next, %hidden.step ]
%hidden.more = icmp ult i32 %f, %hidden br i1 %hidden.more, label %hidden.step, label %hidden.done
hidden.step: %f.wide = zext i32 %f to i64 %unit = add i64 %slot.offset, %f.wide %unit.offset = mul i64 %unit, %length.wide %delta.index = add i64 %delta.base, %unit.offset
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %delta.index %incoming = load float, ptr addrspace(1) %delta.ptr, align 4
%f.row = mul i64 %f.wide, %channels.wide %f.local = add i64 %slice, %f.row %channel.wide = add i64 %channel, 0 %weight.index = add i64 %f.local, %channel.wide
%weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %weight.index %weight.model = load i8, ptr addrspace(1) %weight.ptr, align 1 %weight = call float @recipe.decode(i8 %weight.model)
%product = call float @recipe.state.mul(float %incoming, float %weight) %inner.next = call float @recipe.state.add(float %inner, float %product) %f.next = add i32 %f, 1 br label %hidden.loop
hidden.done: %grown = call float @recipe.state.add(float %total, float %inner) %slot.grown = add i32 %slot, 1 br label %advance
advance: %total.next = phi float [ %grown, %hidden.done ], [ %total, %step ] %slot.next = phi i32 [ %slot.grown, %hidden.done ], [ %slot, %step ]
%e.next = add i32 %e, 1 br label %expert.loop
done: %adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %p %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%sum = call float @recipe.state.add(float %prior, float %total) store float %sum, ptr addrspace(1) %adjoint.ptr, align 4 ret void }
; Weight element %p of a gate or up table sums over the positions routed to its
; own expert, in the bucket order.
define internal void @expert_in_reverse_weight_body( ptr addrspace(1) %input, ptr addrspace(1) %delta, ptr addrspace(1) %context, ptr addrspace(1) %gradient, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %offset ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%plane = mul i64 %hidden.wide, %channels.wide %expert = udiv i64 %p, %plane %local = urem i64 %p, %plane %f = udiv i64 %local, %channels.wide %channel = urem i64 %local, %channels.wide %expert.i32 = trunc i64 %expert to i32
%start = call i32 @moe_bucket_start( ptr addrspace(1) %context, i32 %expert.i32, i32 %experts )
%count.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %expert %count = load i32, ptr addrspace(1) %count.ptr, align 4
%units = mul i64 %top.wide, %hidden.wide %delta.narrow = mul i64 %units, %length.wide %input.narrow = mul i64 %channels.wide, %length.wide
br label %loop
loop: %i = phi i32 [ 0, %entry ], [ %i.next, %step ] %total = phi float [ %state.zero, %entry ], [ %total.next, %step ]
%more = icmp ult i32 %i, %count br i1 %more, label %step, label %done
step: %start.wide = zext i32 %start to i64 %i.wide = zext i32 %i to i64 %read.index = add i64 %start.wide, %i.wide %read.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %read.index %packed = load i32, ptr addrspace(1) %read.ptr, align 4
%packed.wide = zext i32 %packed to i64 %pair = udiv i64 %packed.wide, %top.wide %slot = urem i64 %packed.wide, %top.wide %row = udiv i64 %pair, %length.wide %position = urem i64 %pair, %length.wide
%slot.offset = mul i64 %slot, %hidden.wide %unit = add i64 %slot.offset, %f %unit.offset = mul i64 %unit, %length.wide
%delta.row = mul i64 %row, %delta.narrow %delta.local = add i64 %delta.row, %unit.offset %delta.index = add i64 %delta.local, %position
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %delta.index %incoming = load float, ptr addrspace(1) %delta.ptr, align 4
%input.row = mul i64 %row, %input.narrow %channel.offset = mul i64 %channel, %length.wide %input.local = add i64 %input.row, %channel.offset %input.index = add i64 %input.local, %position
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %input.index %value.model = load i8, ptr addrspace(1) %input.ptr, align 1 %value = call float @recipe.decode(i8 %value.model)
%product = call float @recipe.state.mul(float %incoming, float %value) %total.next = call float @recipe.state.add(float %total, float %product) %i.next = add i32 %i, 1 br label %loop
done: %offset.wide = zext i32 %offset to i64 %store.index = add i64 %offset.wide, %p %store.ptr = getelementptr inbounds float, ptr addrspace(1) %gradient, i64 %store.index store float %total, ptr addrspace(1) %store.ptr, align 4 ret void }
; The bucket of an expert starts after every lower expert's positions.
define internal i32 @moe_bucket_start( ptr addrspace(1) %context, i32 %expert, i32 %experts ) #1 { entry:
br label %loop
loop: %e = phi i32 [ 0, %entry ], [ %e.next, %step ] %base = phi i32 [ %experts, %entry ], [ %base.next, %step ]
%more = icmp ult i32 %e, %expert br i1 %more, label %step, label %done
step: %count.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i32 %e %count = load i32, ptr addrspace(1) %count.ptr, align 4
%base.next = add i32 %base, %count %e.next = add i32 %e, 1 br label %loop
done: ret i32 %base }
; Output element %p sums the selected experts' down projections under their
; routing weights, in ascending expert order.
define internal void @expert_out_forward_body( ptr addrspace(1) %values, ptr addrspace(1) %routing, ptr addrspace(1) %selected, ptr addrspace(1) %weights, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %decode, i32 %lane, i32 %width ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %channel = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
%units = mul i64 %top.wide, %hidden.wide %values.narrow = mul i64 %units, %length.wide %values.row = mul i64 %row, %values.narrow %values.base = add i64 %values.row, %position
%plane = mul i64 %channels.wide, %hidden.wide %channel.offset = mul i64 %channel, %hidden.wide
%selected.row = mul i64 %row, %length.wide %selected.position = add i64 %selected.row, %position %selected.stride = add i64 %top.wide, 1
%selected.base = mul i64 %selected.position, %selected.stride %selected.count.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %selected.base
%selected.count = load i32, ptr addrspace(1) %selected.count.ptr, align 4
br label %expert.loop
expert.loop: %slot = phi i32 [ 0, %entry ], [ %slot.next, %hidden.done ] %total = phi float [ %state.zero, %entry ], [ %total.next, %hidden.done ]
%expert.more = icmp ult i32 %slot, %selected.count br i1 %expert.more, label %step, label %done
step: %slot.wide = zext i32 %slot to i64 %selected.local = add i64 %slot.wide, 1 %selected.index = add i64 %selected.base, %selected.local
%selected.ptr = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %selected.index %e = load i32, ptr addrspace(1) %selected.ptr, align 4
%e.wide = zext i32 %e to i64 %e.offset = mul i64 %e.wide, %length.wide %e.index = add i64 %routing.base, %e.offset
%e.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %e.index %e.weight = load i8, ptr addrspace(1) %e.ptr, align 1
%slice = mul i64 %e.wide, %plane %weight.base = add i64 %slice, %channel.offset %slot.offset = mul i64 %slot.wide, %hidden.wide br label %hidden.loop
hidden.loop: %f = phi i32 [ %lane, %step ], [ %f.next, %hidden.step ] %inner = phi float [ %state.zero, %step ], [ %inner.next, %hidden.step ]
%hidden.more = icmp ult i32 %f, %hidden br i1 %hidden.more, label %hidden.step, label %hidden.done
hidden.step: %f.wide = zext i32 %f to i64 %weight.index = add i64 %weight.base, %f.wide
%weight = call i8 @recipe.model.weight(ptr addrspace(1) %weights, i64 %weight.index, i32 %decode)
%unit = add i64 %slot.offset, %f.wide %unit.offset = mul i64 %unit, %length.wide %value.index = add i64 %values.base, %unit.offset
%value.ptr = getelementptr inbounds i8, ptr addrspace(1) %values, i64 %value.index %value = load i8, ptr addrspace(1) %value.ptr, align 1
%weight.state = call float @recipe.decode(i8 %weight)
%value.state = call float @recipe.decode(i8 %value)
%product = call float @recipe.state.mul(float %weight.state, float %value.state)
%inner.next = call float @recipe.state.add(float %inner, float %product) %f.next = add i32 %f, %width br label %hidden.loop
hidden.done: %sum = call float @expert_wave_sum(float %inner, i32 %lane, i32 %width)
%routing.state = call float @recipe.decode(i8 %e.weight)
%scaled = call float @recipe.state.mul(float %routing.state, float %sum)
%total.next = call float @recipe.state.add(float %total, float %scaled) %slot.next = add i32 %slot, 1 br label %expert.loop
done: %owner = icmp eq i32 %lane, 0 br i1 %owner, label %store, label %exit
store: %result = call i8 @recipe.encode(float %total)
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %p store i8 %result, ptr addrspace(1) %output.ptr, align 1 br label %exit
exit: ret void }
; Hidden element %p of the gated product receives its own expert's column of
; the down table under its routing weight.
define internal void @expert_out_reverse_values_body( ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%units = mul i64 %top.wide, %hidden.wide %narrow = mul i64 %units, %length.wide
%row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %unit = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%slot = udiv i64 %unit, %hidden.wide %f = urem i64 %unit, %hidden.wide %slot.i32 = trunc i64 %slot to i32
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
br label %find.loop
find.loop: %c = phi i32 [ 0, %entry ], [ %c.next, %find.step ] %seen = phi i32 [ 0, %entry ], [ %seen.next, %find.step ] %chosen = phi i32 [ -1, %entry ], [ %chosen.next, %find.step ]
%find.more = icmp ult i32 %c, %experts br i1 %find.more, label %find.step, label %sum.entry
find.step: %c.wide = zext i32 %c to i64 %c.offset = mul i64 %c.wide, %length.wide %c.index = add i64 %routing.base, %c.offset
%c.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %c.index %c.weight = load i8, ptr addrspace(1) %c.ptr, align 1
%c.zero = call i1 @recipe.oeq(i8 %c.weight, i8 0) %c.taken = xor i1 %c.zero, true
%c.match = icmp eq i32 %seen, %slot.i32 %c.unset = icmp eq i32 %chosen, -1 %c.ready = and i1 %c.match, %c.unset %c.pick = and i1 %c.taken, %c.ready
%chosen.next = select i1 %c.pick, i32 %c, i32 %chosen %c.step = zext i1 %c.taken to i32 %seen.next = add i32 %seen, %c.step %c.next = add i32 %c, 1 br label %find.loop
sum.entry: %found = icmp ne i32 %chosen, -1 %expert = select i1 %found, i32 %chosen, i32 0
%expert.wide = zext i32 %expert to i64 %expert.offset = mul i64 %expert.wide, %length.wide %expert.index = add i64 %routing.base, %expert.offset
%expert.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %expert.index %routed.model = load i8, ptr addrspace(1) %expert.ptr, align 1 %routed = call float @recipe.decode(i8 %routed.model)
%plane = mul i64 %channels.wide, %hidden.wide %slice = mul i64 %expert.wide, %plane
%delta.narrow = mul i64 %channels.wide, %length.wide %delta.row = mul i64 %row, %delta.narrow %delta.base = add i64 %delta.row, %position
br label %sum.loop
sum.loop: %o = phi i32 [ 0, %sum.entry ], [ %o.next, %sum.step ] %total = phi float [ %state.zero, %sum.entry ], [ %total.next, %sum.step ]
%sum.more = icmp ult i32 %o, %channels br i1 %sum.more, label %sum.step, label %done
sum.step: %o.wide = zext i32 %o to i64 %o.offset = mul i64 %o.wide, %length.wide %delta.index = add i64 %delta.base, %o.offset
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %delta.index %incoming = load float, ptr addrspace(1) %delta.ptr, align 4
%o.row = mul i64 %o.wide, %hidden.wide %o.local = add i64 %slice, %o.row %weight.index = add i64 %o.local, %f
%weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %weight.index %weight.model = load i8, ptr addrspace(1) %weight.ptr, align 1 %weight = call float @recipe.decode(i8 %weight.model)
%product = call float @recipe.state.mul(float %incoming, float %weight) %total.next = call float @recipe.state.add(float %total, float %product) %o.next = add i32 %o, 1 br label %sum.loop
done: %scaled = call float @recipe.state.mul(float %routed, float %total) %result = select i1 %found, float %scaled, float %state.zero
%adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %p %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%sum = call float @recipe.state.add(float %prior, float %result) store float %sum, ptr addrspace(1) %adjoint.ptr, align 4 ret void }
; Weight element %p of the down table sums over the positions routed to its own
; expert, in the bucket order.
define internal void @expert_out_reverse_weight_body( ptr addrspace(1) %values, ptr addrspace(1) %routing, ptr addrspace(1) %delta, ptr addrspace(1) %context, ptr addrspace(1) %gradient, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %offset ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%plane = mul i64 %channels.wide, %hidden.wide %expert = udiv i64 %p, %plane %local = urem i64 %p, %plane %channel = udiv i64 %local, %hidden.wide %f = urem i64 %local, %hidden.wide %expert.i32 = trunc i64 %expert to i32
%start = call i32 @moe_bucket_start( ptr addrspace(1) %context, i32 %expert.i32, i32 %experts )
%count.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %expert %count = load i32, ptr addrspace(1) %count.ptr, align 4
%units = mul i64 %top.wide, %hidden.wide %values.narrow = mul i64 %units, %length.wide %delta.narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %experts.wide, %length.wide
%expert.offset = mul i64 %expert, %length.wide %channel.offset = mul i64 %channel, %length.wide
br label %loop
loop: %i = phi i32 [ 0, %entry ], [ %i.next, %step ] %total = phi float [ %state.zero, %entry ], [ %total.next, %step ]
%more = icmp ult i32 %i, %count br i1 %more, label %step, label %done
step: %read.index = add i32 %start, %i %read.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i32 %read.index %packed = load i32, ptr addrspace(1) %read.ptr, align 4
%pair = udiv i32 %packed, %top %slot = urem i32 %packed, %top %row = udiv i32 %pair, %length %position = urem i32 %pair, %length
%row.wide = zext i32 %row to i64 %position.wide = zext i32 %position to i64 %routing.row = mul i64 %row.wide, %per.row %routing.local = add i64 %routing.row, %expert.offset %routing.index = add i64 %routing.local, %position.wide
%routing.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %routing.index %routed.model = load i8, ptr addrspace(1) %routing.ptr, align 1 %routed = call float @recipe.decode(i8 %routed.model)
%delta.row = mul i64 %row.wide, %delta.narrow %delta.channel.offset = mul i64 %channel, %length.wide %delta.local = add i64 %delta.row, %delta.channel.offset %delta.index = add i64 %delta.local, %position.wide
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %delta.index %incoming = load float, ptr addrspace(1) %delta.ptr, align 4
%slot.wide = zext i32 %slot to i64 %slot.offset = mul i64 %slot.wide, %hidden.wide %unit = add i64 %slot.offset, %f %unit.offset = mul i64 %unit, %length.wide
%values.row = mul i64 %row.wide, %values.narrow %values.local = add i64 %values.row, %unit.offset %values.index = add i64 %values.local, %position.wide
%values.ptr = getelementptr inbounds i8, ptr addrspace(1) %values, i64 %values.index %value.model = load i8, ptr addrspace(1) %values.ptr, align 1 %value = call float @recipe.decode(i8 %value.model)
%weighted = call float @recipe.state.mul(float %routed, float %incoming) %product = call float @recipe.state.mul(float %weighted, float %value)
%total.next = call float @recipe.state.add(float %total, float %product) %i.next = add i32 %i, 1 br label %loop
done: %offset.wide = zext i32 %offset to i64 %store.index = add i64 %offset.wide, %p %store.ptr = getelementptr inbounds float, ptr addrspace(1) %gradient, i64 %store.index store float %total, ptr addrspace(1) %store.ptr, align 4 ret void }
; Position %p sends each selected expert's down projection back to its own
; routing weight, in ascending expert order.
define internal void @expert_out_reverse_routing_body( ptr addrspace(1) %values, ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top ) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%row = udiv i64 %p, %length.wide %position = urem i64 %p, %length.wide
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
%units = mul i64 %top.wide, %hidden.wide %values.narrow = mul i64 %units, %length.wide %values.row = mul i64 %row, %values.narrow %values.base = add i64 %values.row, %position
%delta.narrow = mul i64 %channels.wide, %length.wide %delta.row = mul i64 %row, %delta.narrow %delta.base = add i64 %delta.row, %position
%plane = mul i64 %channels.wide, %hidden.wide
br label %expert.loop
expert.loop: %e = phi i32 [ 0, %entry ], [ %e.next, %advance ] %slot = phi i32 [ 0, %entry ], [ %slot.next, %advance ]
%expert.more = icmp ult i32 %e, %experts br i1 %expert.more, label %step, label %done
step: %e.wide = zext i32 %e to i64 %e.offset = mul i64 %e.wide, %length.wide %e.index = add i64 %routing.base, %e.offset
%e.ptr = getelementptr inbounds i8, ptr addrspace(1) %routing, i64 %e.index %e.weight = load i8, ptr addrspace(1) %e.ptr, align 1
%e.zero = call i1 @recipe.oeq(i8 %e.weight, i8 0) %e.taken = xor i1 %e.zero, true
br i1 %e.taken, label %channel.entry, label %advance
channel.entry: %slice = mul i64 %e.wide, %plane %slot.wide = zext i32 %slot to i64 %slot.offset = mul i64 %slot.wide, %hidden.wide br label %channel.loop
channel.loop: %o = phi i32 [ 0, %channel.entry ], [ %o.next, %channel.done ] %outer = phi float [ %state.zero, %channel.entry ], [ %outer.next, %channel.done ]
%channel.more = icmp ult i32 %o, %channels br i1 %channel.more, label %channel.step, label %write
channel.step: %o.wide = zext i32 %o to i64 %o.offset = mul i64 %o.wide, %length.wide %delta.index = add i64 %delta.base, %o.offset
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %delta.index %incoming = load float, ptr addrspace(1) %delta.ptr, align 4
%o.row = mul i64 %o.wide, %hidden.wide %weight.base = add i64 %slice, %o.row br label %hidden.loop
hidden.loop: %f = phi i32 [ 0, %channel.step ], [ %f.next, %hidden.step ] %inner = phi float [ %state.zero, %channel.step ], [ %inner.next, %hidden.step ]
%hidden.more = icmp ult i32 %f, %hidden br i1 %hidden.more, label %hidden.step, label %channel.done
hidden.step: %f.wide = zext i32 %f to i64 %weight.index = add i64 %weight.base, %f.wide %weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %weight.index
%weight.model = load i8, ptr addrspace(1) %weight.ptr, align 1 %weight = call float @recipe.decode(i8 %weight.model)
%unit = add i64 %slot.offset, %f.wide %unit.offset = mul i64 %unit, %length.wide %value.index = add i64 %values.base, %unit.offset
%value.ptr = getelementptr inbounds i8, ptr addrspace(1) %values, i64 %value.index %value.model = load i8, ptr addrspace(1) %value.ptr, align 1 %value = call float @recipe.decode(i8 %value.model)
%product = call float @recipe.state.mul(float %weight, float %value) %inner.next = call float @recipe.state.add(float %inner, float %product) %f.next = add i32 %f, 1 br label %hidden.loop
channel.done: %term = call float @recipe.state.mul(float %incoming, float %inner) %outer.next = call float @recipe.state.add(float %outer, float %term) %o.next = add i32 %o, 1 br label %channel.loop
write: %adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %adjoint, i64 %e.index %prior = load float, ptr addrspace(1) %adjoint.ptr, align 4
%sum = call float @recipe.state.add(float %prior, float %outer) store float %sum, ptr addrspace(1) %adjoint.ptr, align 4
%slot.grown = add i32 %slot, 1 br label %advance
advance: %slot.next = phi i32 [ %slot.grown, %write ], [ %slot, %step ] %e.next = add i32 %e, 1 br label %expert.loop
done: ret void }
define internal i8 @sigmoid(i8 %x) #1 { entry: %negative = call i8 @recipe.neg(i8 %x)
%exponential = call i8 @recipe.exp(i8 %negative) %denominator = call i8 @recipe.add(i8 56, i8 %exponential)
%value = call i8 @recipe.div(i8 56, i8 %denominator) ret i8 %value }
define internal float @attention_tile_dot(i32 %left, i32 %right, i32 %width, i32 %left.base, i32 %right.base) #1 { entry:
%zero = call float @recipe.state.from.u1(i1 false)
br label %channel.loop
channel.loop:
%channel = phi i32 [ 0, %entry ], [ %channel.next, %channel.step ]
%sum = phi float [ %zero, %entry ], [ %sum.next, %channel.step ]
%more = icmp ult i32 %channel, %width
br i1 %more, label %channel.step, label %done
channel.step:
%left.row = mul i32 %left, %width
%left.local = add i32 %left.row, %channel
%left.index = add i32 %left.base, %left.local
%left.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %left.index
%left.value = load i8, ptr addrspace(3) %left.ptr, align 1
%right.row = mul i32 %right, %width
%right.local = add i32 %right.row, %channel
%right.index = add i32 %right.base, %right.local
%right.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %right.index
%right.value = load i8, ptr addrspace(3) %right.ptr, align 1
%left.wide = call float @recipe.decode(i8 %left.value)
%right.wide = call float @recipe.decode(i8 %right.value)
%product = call float @recipe.state.mul(float %left.wide, float %right.wide)
%sum.next = call float @recipe.state.add(float %sum, float %product)
%channel.next = add i32 %channel, 1
br label %channel.loop
done:
ret float %sum
}
define internal float @attention_tile_dot_state(i32 %left, i32 %right, i32 %width, i32 %left.base, i32 %right.base) #1 { entry:
%zero = call float @recipe.state.from.u1(i1 false)
br label %channel.loop
channel.loop:
%channel = phi i32 [ 0, %entry ], [ %channel.next, %channel.step ]
%sum = phi float [ %zero, %entry ], [ %sum.next, %channel.step ]
%more = icmp ult i32 %channel, %width
br i1 %more, label %channel.step, label %done
channel.step:
%left.row = mul i32 %left, %width
%left.local = add i32 %left.row, %channel
%left.index = add i32 %left.base, %left.local
%left.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %left.index
%left.wide = load float, ptr addrspace(3) %left.ptr, align 4
%right.row = mul i32 %right, %width
%right.local = add i32 %right.row, %channel
%right.index = add i32 %right.base, %right.local
%right.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %right.index
%right.wide = load float, ptr addrspace(3) %right.ptr, align 4
%product = call float @recipe.state.mul(float %left.wide, float %right.wide)
%sum.next = call float @recipe.state.add(float %sum, float %product)
%channel.next = add i32 %channel, 1
br label %channel.loop
done:
ret float %sum
}
define internal i8 @attention_tile_score(i32 %query, i32 %key, i32 %width, i32 %key.base, i8 %scale) #1 { entry:
%sum = call float @attention_tile_dot(i32 %query, i32 %key, i32 %width, i32 0, i32 %key.base)
%scale.wide = call float @recipe.decode(i8 %scale)
%score = call float @recipe.state.div(float %sum, float %scale.wide)
%result = call i8 @recipe.encode(float %score)
ret i8 %result
}
define internal void @attention_step_score_store(ptr addrspace(1) %context, i64 %index, float %value) #1 { entry:
%ptr = getelementptr float, ptr addrspace(1) %context, i64 %index
store float %value, ptr addrspace(1) %ptr, align 4
ret void
}
define internal float @attention_step_score_load(ptr addrspace(1) %context, i64 %index) #1 { entry:
%ptr = getelementptr float, ptr addrspace(1) %context, i64 %index
%value = load float, ptr addrspace(1) %ptr, align 4
ret float %value
}
; Whole-grid inference attention for one query. The ordinary context's two
; statistics planes temporarily hold one packed state float per score; the
; dedicated K/V context keeps the settled history in the cache type.
define internal float @attention_step_key_dot(ptr addrspace(3) %query, ptr addrspace(1) %kv.context, i32 %key, i32 %kv.head, i32 %width, i32 %length) #1 {
entry:
%kv.channel.base = mul i32 %kv.head, %width
br label %loop
loop:
%channel = phi i32 [ 0, %entry ], [ %channel.next, %step ]
%sum.0 = phi float [ 0x0000000000000000, %entry ], [ %sum.0.next, %step ]
%sum.1 = phi float [ 0x0000000000000000, %entry ], [ %sum.1.next, %step ]
%sum.2 = phi float [ 0x0000000000000000, %entry ], [ %sum.2.next, %step ]
%sum.3 = phi float [ 0x0000000000000000, %entry ], [ %sum.3.next, %step ]
%sum.4 = phi float [ 0x0000000000000000, %entry ], [ %sum.4.next, %step ]
%sum.5 = phi float [ 0x0000000000000000, %entry ], [ %sum.5.next, %step ]
%sum.6 = phi float [ 0x0000000000000000, %entry ], [ %sum.6.next, %step ]
%sum.7 = phi float [ 0x0000000000000000, %entry ], [ %sum.7.next, %step ]
%sum.8 = phi float [ 0x0000000000000000, %entry ], [ %sum.8.next, %step ]
%sum.9 = phi float [ 0x0000000000000000, %entry ], [ %sum.9.next, %step ]
%sum.10 = phi float [ 0x0000000000000000, %entry ], [ %sum.10.next, %step ]
%sum.11 = phi float [ 0x0000000000000000, %entry ], [ %sum.11.next, %step ]
%sum.12 = phi float [ 0x0000000000000000, %entry ], [ %sum.12.next, %step ]
%sum.13 = phi float [ 0x0000000000000000, %entry ], [ %sum.13.next, %step ]
%sum.14 = phi float [ 0x0000000000000000, %entry ], [ %sum.14.next, %step ]
%sum.15 = phi float [ 0x0000000000000000, %entry ], [ %sum.15.next, %step ]
%more = icmp ult i32 %channel, %width
br i1 %more, label %step, label %done
step:
%channel.0 = add i32 %channel, 0
%channel.1 = add i32 %channel, 1
%channel.2 = add i32 %channel, 2
%channel.3 = add i32 %channel, 3
%channel.4 = add i32 %channel, 4
%channel.5 = add i32 %channel, 5
%channel.6 = add i32 %channel, 6
%channel.7 = add i32 %channel, 7
%channel.8 = add i32 %channel, 8
%channel.9 = add i32 %channel, 9
%channel.10 = add i32 %channel, 10
%channel.11 = add i32 %channel, 11
%channel.12 = add i32 %channel, 12
%channel.13 = add i32 %channel, 13
%channel.14 = add i32 %channel, 14
%channel.15 = add i32 %channel, 15
%active.0 = icmp ult i32 %channel.0, %width
%active.1 = icmp ult i32 %channel.1, %width
%active.2 = icmp ult i32 %channel.2, %width
%active.3 = icmp ult i32 %channel.3, %width
%active.4 = icmp ult i32 %channel.4, %width
%active.5 = icmp ult i32 %channel.5, %width
%active.6 = icmp ult i32 %channel.6, %width
%active.7 = icmp ult i32 %channel.7, %width
%active.8 = icmp ult i32 %channel.8, %width
%active.9 = icmp ult i32 %channel.9, %width
%active.10 = icmp ult i32 %channel.10, %width
%active.11 = icmp ult i32 %channel.11, %width
%active.12 = icmp ult i32 %channel.12, %width
%active.13 = icmp ult i32 %channel.13, %width
%active.14 = icmp ult i32 %channel.14, %width
%active.15 = icmp ult i32 %channel.15, %width
%safe.0 = select i1 %active.0, i32 %channel.0, i32 0
%safe.1 = select i1 %active.1, i32 %channel.1, i32 0
%safe.2 = select i1 %active.2, i32 %channel.2, i32 0
%safe.3 = select i1 %active.3, i32 %channel.3, i32 0
%safe.4 = select i1 %active.4, i32 %channel.4, i32 0
%safe.5 = select i1 %active.5, i32 %channel.5, i32 0
%safe.6 = select i1 %active.6, i32 %channel.6, i32 0
%safe.7 = select i1 %active.7, i32 %channel.7, i32 0
%safe.8 = select i1 %active.8, i32 %channel.8, i32 0
%safe.9 = select i1 %active.9, i32 %channel.9, i32 0
%safe.10 = select i1 %active.10, i32 %channel.10, i32 0
%safe.11 = select i1 %active.11, i32 %channel.11, i32 0
%safe.12 = select i1 %active.12, i32 %channel.12, i32 0
%safe.13 = select i1 %active.13, i32 %channel.13, i32 0
%safe.14 = select i1 %active.14, i32 %channel.14, i32 0
%safe.15 = select i1 %active.15, i32 %channel.15, i32 0
%q.0.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.0
%q.1.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.1
%q.2.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.2
%q.3.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.3
%q.4.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.4
%q.5.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.5
%q.6.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.6
%q.7.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.7
%q.8.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.8
%q.9.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.9
%q.10.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.10
%q.11.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.11
%q.12.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.12
%q.13.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.13
%q.14.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.14
%q.15.ptr = getelementptr float, ptr addrspace(3) %query, i32 %safe.15
%q.0.raw = load float, ptr addrspace(3) %q.0.ptr, align 4
%q.1.raw = load float, ptr addrspace(3) %q.1.ptr, align 4
%q.2.raw = load float, ptr addrspace(3) %q.2.ptr, align 4
%q.3.raw = load float, ptr addrspace(3) %q.3.ptr, align 4
%q.4.raw = load float, ptr addrspace(3) %q.4.ptr, align 4
%q.5.raw = load float, ptr addrspace(3) %q.5.ptr, align 4
%q.6.raw = load float, ptr addrspace(3) %q.6.ptr, align 4
%q.7.raw = load float, ptr addrspace(3) %q.7.ptr, align 4
%q.8.raw = load float, ptr addrspace(3) %q.8.ptr, align 4
%q.9.raw = load float, ptr addrspace(3) %q.9.ptr, align 4
%q.10.raw = load float, ptr addrspace(3) %q.10.ptr, align 4
%q.11.raw = load float, ptr addrspace(3) %q.11.ptr, align 4
%q.12.raw = load float, ptr addrspace(3) %q.12.ptr, align 4
%q.13.raw = load float, ptr addrspace(3) %q.13.ptr, align 4
%q.14.raw = load float, ptr addrspace(3) %q.14.ptr, align 4
%q.15.raw = load float, ptr addrspace(3) %q.15.ptr, align 4
%q.0 = select i1 %active.0, float %q.0.raw, float 0x0000000000000000
%q.1 = select i1 %active.1, float %q.1.raw, float 0x0000000000000000
%q.2 = select i1 %active.2, float %q.2.raw, float 0x0000000000000000
%q.3 = select i1 %active.3, float %q.3.raw, float 0x0000000000000000
%q.4 = select i1 %active.4, float %q.4.raw, float 0x0000000000000000
%q.5 = select i1 %active.5, float %q.5.raw, float 0x0000000000000000
%q.6 = select i1 %active.6, float %q.6.raw, float 0x0000000000000000
%q.7 = select i1 %active.7, float %q.7.raw, float 0x0000000000000000
%q.8 = select i1 %active.8, float %q.8.raw, float 0x0000000000000000
%q.9 = select i1 %active.9, float %q.9.raw, float 0x0000000000000000
%q.10 = select i1 %active.10, float %q.10.raw, float 0x0000000000000000
%q.11 = select i1 %active.11, float %q.11.raw, float 0x0000000000000000
%q.12 = select i1 %active.12, float %q.12.raw, float 0x0000000000000000
%q.13 = select i1 %active.13, float %q.13.raw, float 0x0000000000000000
%q.14 = select i1 %active.14, float %q.14.raw, float 0x0000000000000000
%q.15 = select i1 %active.15, float %q.15.raw, float 0x0000000000000000
%key.0.channel = add i32 %kv.channel.base, %safe.0
%key.1.channel = add i32 %kv.channel.base, %safe.1
%key.2.channel = add i32 %kv.channel.base, %safe.2
%key.3.channel = add i32 %kv.channel.base, %safe.3
%key.4.channel = add i32 %kv.channel.base, %safe.4
%key.5.channel = add i32 %kv.channel.base, %safe.5
%key.6.channel = add i32 %kv.channel.base, %safe.6
%key.7.channel = add i32 %kv.channel.base, %safe.7
%key.8.channel = add i32 %kv.channel.base, %safe.8
%key.9.channel = add i32 %kv.channel.base, %safe.9
%key.10.channel = add i32 %kv.channel.base, %safe.10
%key.11.channel = add i32 %kv.channel.base, %safe.11
%key.12.channel = add i32 %kv.channel.base, %safe.12
%key.13.channel = add i32 %kv.channel.base, %safe.13
%key.14.channel = add i32 %kv.channel.base, %safe.14
%key.15.channel = add i32 %kv.channel.base, %safe.15
%key.0.base = mul i32 %key.0.channel, %length
%key.1.base = mul i32 %key.1.channel, %length
%key.2.base = mul i32 %key.2.channel, %length
%key.3.base = mul i32 %key.3.channel, %length
%key.4.base = mul i32 %key.4.channel, %length
%key.5.base = mul i32 %key.5.channel, %length
%key.6.base = mul i32 %key.6.channel, %length
%key.7.base = mul i32 %key.7.channel, %length
%key.8.base = mul i32 %key.8.channel, %length
%key.9.base = mul i32 %key.9.channel, %length
%key.10.base = mul i32 %key.10.channel, %length
%key.11.base = mul i32 %key.11.channel, %length
%key.12.base = mul i32 %key.12.channel, %length
%key.13.base = mul i32 %key.13.channel, %length
%key.14.base = mul i32 %key.14.channel, %length
%key.15.base = mul i32 %key.15.channel, %length
%key.0.index = add i32 %key.0.base, %key
%key.1.index = add i32 %key.1.base, %key
%key.2.index = add i32 %key.2.base, %key
%key.3.index = add i32 %key.3.base, %key
%key.4.index = add i32 %key.4.base, %key
%key.5.index = add i32 %key.5.base, %key
%key.6.index = add i32 %key.6.base, %key
%key.7.index = add i32 %key.7.base, %key
%key.8.index = add i32 %key.8.base, %key
%key.9.index = add i32 %key.9.base, %key
%key.10.index = add i32 %key.10.base, %key
%key.11.index = add i32 %key.11.base, %key
%key.12.index = add i32 %key.12.base, %key
%key.13.index = add i32 %key.13.base, %key
%key.14.index = add i32 %key.14.base, %key
%key.15.index = add i32 %key.15.base, %key
%key.0.wide = zext i32 %key.0.index to i64
%key.1.wide = zext i32 %key.1.index to i64
%key.2.wide = zext i32 %key.2.index to i64
%key.3.wide = zext i32 %key.3.index to i64
%key.4.wide = zext i32 %key.4.index to i64
%key.5.wide = zext i32 %key.5.index to i64
%key.6.wide = zext i32 %key.6.index to i64
%key.7.wide = zext i32 %key.7.index to i64
%key.8.wide = zext i32 %key.8.index to i64
%key.9.wide = zext i32 %key.9.index to i64
%key.10.wide = zext i32 %key.10.index to i64
%key.11.wide = zext i32 %key.11.index to i64
%key.12.wide = zext i32 %key.12.index to i64
%key.13.wide = zext i32 %key.13.index to i64
%key.14.wide = zext i32 %key.14.index to i64
%key.15.wide = zext i32 %key.15.index to i64
%key.0.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.0.wide
%key.1.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.1.wide
%key.2.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.2.wide
%key.3.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.3.wide
%key.4.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.4.wide
%key.5.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.5.wide
%key.6.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.6.wide
%key.7.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.7.wide
%key.8.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.8.wide
%key.9.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.9.wide
%key.10.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.10.wide
%key.11.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.11.wide
%key.12.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.12.wide
%key.13.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.13.wide
%key.14.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.14.wide
%key.15.ptr = getelementptr i8, ptr addrspace(1) %kv.context, i64 %key.15.wide
%key.0.load = load i8, ptr addrspace(1) %key.0.ptr, align 1
%key.1.load = load i8, ptr addrspace(1) %key.1.ptr, align 1
%key.2.load = load i8, ptr addrspace(1) %key.2.ptr, align 1
%key.3.load = load i8, ptr addrspace(1) %key.3.ptr, align 1
%key.4.load = load i8, ptr addrspace(1) %key.4.ptr, align 1
%key.5.load = load i8, ptr addrspace(1) %key.5.ptr, align 1
%key.6.load = load i8, ptr addrspace(1) %key.6.ptr, align 1
%key.7.load = load i8, ptr addrspace(1) %key.7.ptr, align 1
%key.8.load = load i8, ptr addrspace(1) %key.8.ptr, align 1
%key.9.load = load i8, ptr addrspace(1) %key.9.ptr, align 1
%key.10.load = load i8, ptr addrspace(1) %key.10.ptr, align 1
%key.11.load = load i8, ptr addrspace(1) %key.11.ptr, align 1
%key.12.load = load i8, ptr addrspace(1) %key.12.ptr, align 1
%key.13.load = load i8, ptr addrspace(1) %key.13.ptr, align 1
%key.14.load = load i8, ptr addrspace(1) %key.14.ptr, align 1
%key.15.load = load i8, ptr addrspace(1) %key.15.ptr, align 1
%key.0.raw = call float @recipe.kv.to.state(i8 %key.0.load)
%key.1.raw = call float @recipe.kv.to.state(i8 %key.1.load)
%key.2.raw = call float @recipe.kv.to.state(i8 %key.2.load)
%key.3.raw = call float @recipe.kv.to.state(i8 %key.3.load)
%key.4.raw = call float @recipe.kv.to.state(i8 %key.4.load)
%key.5.raw = call float @recipe.kv.to.state(i8 %key.5.load)
%key.6.raw = call float @recipe.kv.to.state(i8 %key.6.load)
%key.7.raw = call float @recipe.kv.to.state(i8 %key.7.load)
%key.8.raw = call float @recipe.kv.to.state(i8 %key.8.load)
%key.9.raw = call float @recipe.kv.to.state(i8 %key.9.load)
%key.10.raw = call float @recipe.kv.to.state(i8 %key.10.load)
%key.11.raw = call float @recipe.kv.to.state(i8 %key.11.load)
%key.12.raw = call float @recipe.kv.to.state(i8 %key.12.load)
%key.13.raw = call float @recipe.kv.to.state(i8 %key.13.load)
%key.14.raw = call float @recipe.kv.to.state(i8 %key.14.load)
%key.15.raw = call float @recipe.kv.to.state(i8 %key.15.load)
%key.0 = select i1 %active.0, float %key.0.raw, float 0x0000000000000000
%key.1 = select i1 %active.1, float %key.1.raw, float 0x0000000000000000
%key.2 = select i1 %active.2, float %key.2.raw, float 0x0000000000000000
%key.3 = select i1 %active.3, float %key.3.raw, float 0x0000000000000000
%key.4 = select i1 %active.4, float %key.4.raw, float 0x0000000000000000
%key.5 = select i1 %active.5, float %key.5.raw, float 0x0000000000000000
%key.6 = select i1 %active.6, float %key.6.raw, float 0x0000000000000000
%key.7 = select i1 %active.7, float %key.7.raw, float 0x0000000000000000
%key.8 = select i1 %active.8, float %key.8.raw, float 0x0000000000000000
%key.9 = select i1 %active.9, float %key.9.raw, float 0x0000000000000000
%key.10 = select i1 %active.10, float %key.10.raw, float 0x0000000000000000
%key.11 = select i1 %active.11, float %key.11.raw, float 0x0000000000000000
%key.12 = select i1 %active.12, float %key.12.raw, float 0x0000000000000000
%key.13 = select i1 %active.13, float %key.13.raw, float 0x0000000000000000
%key.14 = select i1 %active.14, float %key.14.raw, float 0x0000000000000000
%key.15 = select i1 %active.15, float %key.15.raw, float 0x0000000000000000
%sum.0.next = call float @recipe.state.madd(float %sum.0, float %q.0, float %key.0)
%sum.1.next = call float @recipe.state.madd(float %sum.1, float %q.1, float %key.1)
%sum.2.next = call float @recipe.state.madd(float %sum.2, float %q.2, float %key.2)
%sum.3.next = call float @recipe.state.madd(float %sum.3, float %q.3, float %key.3)
%sum.4.next = call float @recipe.state.madd(float %sum.4, float %q.4, float %key.4)
%sum.5.next = call float @recipe.state.madd(float %sum.5, float %q.5, float %key.5)
%sum.6.next = call float @recipe.state.madd(float %sum.6, float %q.6, float %key.6)
%sum.7.next = call float @recipe.state.madd(float %sum.7, float %q.7, float %key.7)
%sum.8.next = call float @recipe.state.madd(float %sum.8, float %q.8, float %key.8)
%sum.9.next = call float @recipe.state.madd(float %sum.9, float %q.9, float %key.9)
%sum.10.next = call float @recipe.state.madd(float %sum.10, float %q.10, float %key.10)
%sum.11.next = call float @recipe.state.madd(float %sum.11, float %q.11, float %key.11)
%sum.12.next = call float @recipe.state.madd(float %sum.12, float %q.12, float %key.12)
%sum.13.next = call float @recipe.state.madd(float %sum.13, float %q.13, float %key.13)
%sum.14.next = call float @recipe.state.madd(float %sum.14, float %q.14, float %key.14)
%sum.15.next = call float @recipe.state.madd(float %sum.15, float %q.15, float %key.15)
%channel.next = add i32 %channel, 16
br label %loop
done:
%sum.89 = call float @recipe.state.add(float %sum.8, float %sum.9)
%sum.1011 = call float @recipe.state.add(float %sum.10, float %sum.11)
%sum.1213 = call float @recipe.state.add(float %sum.12, float %sum.13)
%sum.1415 = call float @recipe.state.add(float %sum.14, float %sum.15)
%sum.89.1011 = call float @recipe.state.add(float %sum.89, float %sum.1011)
%sum.1213.1415 = call float @recipe.state.add(float %sum.1213, float %sum.1415)
%sum.8to15 = call float @recipe.state.add(float %sum.89.1011, float %sum.1213.1415)
%sum.01 = call float @recipe.state.add(float %sum.0, float %sum.1)
%sum.23 = call float @recipe.state.add(float %sum.2, float %sum.3)
%sum.45 = call float @recipe.state.add(float %sum.4, float %sum.5)
%sum.67 = call float @recipe.state.add(float %sum.6, float %sum.7)
%sum.0123 = call float @recipe.state.add(float %sum.01, float %sum.23)
%sum.4567 = call float @recipe.state.add(float %sum.45, float %sum.67)
%sum.0to7 = call float @recipe.state.add(float %sum.0123, float %sum.4567)
%sum = call float @recipe.state.add(float %sum.0to7, float %sum.8to15)
ret float %sum
}
define internal i64 @recipe.window.index(i64 %index, i32 %length, i32 %pitch, i32 %origin) #1 {
entry:
	%length.wide = zext i32 %length to i64
	%pitch.wide = zext i32 %pitch to i64
	%origin.wide = zext i32 %origin to i64
	%channel = udiv i64 %index, %length.wide
	%position = urem i64 %index, %length.wide
	%channel.scaled = mul i64 %channel, %pitch.wide
	%position.shifted = sub i64 %position, %origin.wide
	%result = add i64 %channel.scaled, %position.shifted
	ret i64 %result
}
define internal void @attention_forward_step_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture writeonly %output,
ptr addrspace(1) %context, ptr addrspace(1) %kv.context,
i32 %from, i32 %heads, i32 %channels, i32 %position, i32 %kv.heads, i32 %threads, i32 %buffer.length, i32 %buffer.origin) #3 {
entry:
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%global = mul i32 %group, %block
%global.id = add i32 %global, %lid
%length = udiv i32 %from, %channels
%width = udiv i32 %channels, %heads
%kv.group = udiv i32 %heads, %kv.heads
%kv.channels = mul i32 %kv.heads, %width
%kv.plane = mul i32 %kv.channels, %length
%reached = add i32 %position, 1
%wave.width = call i32 @recipe.wavefront.width()
%wave = udiv i32 %lid, %wave.width
%lane = urem i32 %lid, %wave.width
%waves = udiv i32 %block, %wave.width
%global.wave = udiv i32 %global.id, %wave.width
%global.waves = udiv i32 %threads, %wave.width
%width.float = call float @recipe.state.from.u32(i32 %width)
%scale = call float @recipe.state.sqrt(float %width.float)
; The running maximum starts below every score: minus infinity from the
; state's own division, so no literal names a type.
%floor.one = call float @recipe.state.from.u1(i1 true)
%floor.zero = call float @recipe.state.from.u1(i1 false)
%floor.minus = call float @recipe.state.neg(float %floor.one)
%score.floor = call float @recipe.state.div(float %floor.minus, float %floor.zero)
%wave.half.start = lshr i32 %wave.width, 1
%maximum.global.index = add i32 %waves, 0
%denominator.global.index = add i32 %waves, 1
%tile.base = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 0
%tile.float = bitcast ptr addrspace(3) %tile.base to ptr addrspace(3)
%maximum.global.ptr = getelementptr float, ptr addrspace(3) %tile.float, i32 %maximum.global.index
%denominator.global.ptr = getelementptr float, ptr addrspace(3) %tile.float, i32 %denominator.global.index
br label %cache.loop
cache.loop:
%cache.channel = phi i32 [ %global.id, %entry ], [ %cache.channel.next, %cache.step ]
%cache.more = icmp ult i32 %cache.channel, %kv.channels
br i1 %cache.more, label %cache.step, label %cache.done
cache.step:
%cache.channel.base = mul i32 %cache.channel, %length
%cache.position.index = add i32 %cache.channel.base, %position
%cache.position.wide = zext i32 %cache.position.index to i64
%cache.key.source.index = add i32 %from, %cache.position.index
%cache.key.source.wide = zext i32 %cache.key.source.index to i64
%cache.key.source.phys = call i64 @recipe.window.index(i64 %cache.key.source.wide, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%cache.key.source.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %cache.key.source.phys
%cache.key.value = load i8, ptr addrspace(1) %cache.key.source.ptr, align 1
%cache.value.base = add i32 %from, %kv.plane
%cache.value.source.index = add i32 %cache.value.base, %cache.position.index
%cache.value.source.wide = zext i32 %cache.value.source.index to i64
%cache.value.source.phys = call i64 @recipe.window.index(i64 %cache.value.source.wide, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%cache.value.source.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %cache.value.source.phys
%cache.value.value = load i8, ptr addrspace(1) %cache.value.source.ptr, align 1
%cache.key.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %cache.position.wide
%cache.key.kv = call i8 @recipe.kv.encode(i8 %cache.key.value)
store i8 %cache.key.kv, ptr addrspace(1) %cache.key.ptr, align 1
%cache.value.index = add i32 %kv.plane, %cache.position.index
%cache.value.wide = zext i32 %cache.value.index to i64
%cache.value.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %cache.value.wide
%cache.value.kv = call i8 @recipe.kv.encode(i8 %cache.value.value)
store i8 %cache.value.kv, ptr addrspace(1) %cache.value.ptr, align 1
%cache.channel.next = add i32 %cache.channel, %threads
br label %cache.loop
cache.done:
call void @grid_barrier(i32 %threads)
br label %score.tile.loop
score.tile.loop:
%score.tile.base = phi i32 [ 0, %cache.done ], [ %score.tile.next, %score.tile.advance ]
%score.group.active = icmp ult i32 %group, %heads
%score.tile.limit = icmp ult i32 %score.tile.base, %reached
%score.tile.more = and i1 %score.group.active, %score.tile.limit
br i1 %score.tile.more, label %score.query.copy.loop, label %score.done
score.query.copy.loop:
%score.query.channel = phi i32 [ %lid, %score.tile.loop ], [ %score.query.channel.next, %score.query.copy.step ]
%score.query.channel.more = icmp ult i32 %score.query.channel, %width
br i1 %score.query.channel.more, label %score.query.copy.step, label %score.query.copy.done
score.query.copy.step:
%score.head.channel = mul i32 %group, %width
%score.query.global.channel = add i32 %score.head.channel, %score.query.channel
%score.query.channel.base = mul i32 %score.query.global.channel, %length
%score.query.index = add i32 %score.query.channel.base, %position
%score.query.wide = zext i32 %score.query.index to i64
%score.query.phys = call i64 @recipe.window.index(i64 %score.query.wide, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%score.query.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %score.query.phys
%score.query.value = load i8, ptr addrspace(1) %score.query.ptr, align 1
%score.query.float = call float @recipe.state.from.model(i8 %score.query.value)
%score.query.shared.ptr = getelementptr float, ptr addrspace(3) %tile.float, i32 %score.query.channel
store float %score.query.float, ptr addrspace(3) %score.query.shared.ptr, align 4
%score.query.channel.next = add i32 %score.query.channel, %block
br label %score.query.copy.loop
score.query.copy.done:
call void @llvm.nvvm.barrier0()
%score.key = add i32 %score.tile.base, %lid
%score.key.more = icmp ult i32 %score.key, %reached
br i1 %score.key.more, label %score.key.compute, label %score.key.done
score.key.compute:
%score.kv.head = udiv i32 %group, %kv.group
%score.scaled.raw = call float @attention_step_key_dot(ptr addrspace(3) %tile.float, ptr addrspace(1) %kv.context, i32 %score.key, i32 %score.kv.head, i32 %width, i32 %length)
%score.scaled = call float @recipe.state.div(float %score.scaled.raw, float %scale)
%score.row = mul i32 %group, %length
%score.slot = add i32 %score.row, %score.key
%score.slot.wide = zext i32 %score.slot to i64
call void @attention_step_score_store(ptr addrspace(1) %context, i64 %score.slot.wide, float %score.scaled)
br label %score.key.done
score.key.done:
call void @llvm.nvvm.barrier0()
br label %score.tile.advance
score.tile.advance:
%score.tile.next = add i32 %score.tile.base, %block
br label %score.tile.loop
score.done:
call void @grid_barrier(i32 %threads)
%active = icmp ult i32 %group, %heads
%active.limit = select i1 %active, i32 %reached, i32 0
br label %maximum.loop
maximum.loop:
%maximum.key = phi i32 [ %lid, %score.done ], [ %maximum.key.next, %maximum.step ]
%maximum.value = phi float [ %score.floor, %score.done ], [ %maximum.next, %maximum.step ]
%maximum.more = icmp ult i32 %maximum.key, %active.limit
br i1 %maximum.more, label %maximum.step, label %maximum.wave.loop
maximum.step:
%maximum.row = mul i32 %group, %length
%maximum.slot = add i32 %maximum.row, %maximum.key
%maximum.slot.wide = zext i32 %maximum.slot to i64
%maximum.score = call float @attention_step_score_load(ptr addrspace(1) %context, i64 %maximum.slot.wide)
%maximum.larger = call i1 @recipe.state.ogt(float %maximum.score, float %maximum.value)
%maximum.next = select i1 %maximum.larger, float %maximum.score, float %maximum.value
%maximum.key.next = add i32 %maximum.key, %block
br label %maximum.loop
maximum.wave.loop:
%maximum.offset = phi i32 [ %wave.half.start, %maximum.loop ], [ %maximum.offset.next, %maximum.wave.step ]
%maximum.wave.value = phi float [ %maximum.value, %maximum.loop ], [ %maximum.wave.value.next, %maximum.wave.step ]
%maximum.wave.more = icmp ne i32 %maximum.offset, 0
br i1 %maximum.wave.more, label %maximum.wave.step, label %maximum.wave.done
maximum.wave.step:
%maximum.partner.lane = xor i32 %lane, %maximum.offset
%maximum.partner.index = mul i32 %maximum.partner.lane, 4
%maximum.partner = call float @recipe.wave.partner(float %maximum.wave.value, i32 %maximum.partner.index)
%maximum.partner.larger = call i1 @recipe.state.ogt(float %maximum.partner, float %maximum.wave.value)
%maximum.wave.value.next = select i1 %maximum.partner.larger, float %maximum.partner, float %maximum.wave.value
%maximum.offset.next = lshr i32 %maximum.offset, 1
br label %maximum.wave.loop
maximum.wave.done:
%maximum.owner = icmp eq i32 %lane, 0
br i1 %maximum.owner, label %maximum.wave.store, label %maximum.wave.skip
maximum.wave.store:
%maximum.wave.ptr = getelementptr float, ptr addrspace(3) %tile.float, i32 %wave
store float %maximum.wave.value, ptr addrspace(3) %maximum.wave.ptr, align 4
br label %maximum.wave.skip
maximum.wave.skip:
call void @llvm.nvvm.barrier0()
%maximum.group.owner = icmp eq i32 %lid, 0
br i1 %maximum.group.owner, label %maximum.group.loop, label %maximum.group.skip
maximum.group.loop:
%maximum.wave.index = phi i32 [ 0, %maximum.wave.skip ], [ %maximum.wave.index.next, %maximum.group.step ]
%maximum.group.value = phi float [ %score.floor, %maximum.wave.skip ], [ %maximum.group.next, %maximum.group.step ]
%maximum.group.more = icmp ult i32 %maximum.wave.index, %waves
br i1 %maximum.group.more, label %maximum.group.step, label %maximum.group.done
maximum.group.step:
%maximum.group.ptr = getelementptr float, ptr addrspace(3) %tile.float, i32 %maximum.wave.index
%maximum.group.wave = load float, ptr addrspace(3) %maximum.group.ptr, align 4
%maximum.group.larger = call i1 @recipe.state.ogt(float %maximum.group.wave, float %maximum.group.value)
%maximum.group.next = select i1 %maximum.group.larger, float %maximum.group.wave, float %maximum.group.value
%maximum.wave.index.next = add i32 %maximum.wave.index, 1
br label %maximum.group.loop
maximum.group.done:
store float %maximum.group.value, ptr addrspace(3) %maximum.global.ptr, align 4
br label %maximum.group.skip
maximum.group.skip:
call void @llvm.nvvm.barrier0()
%maximum.global = load float, ptr addrspace(3) %maximum.global.ptr, align 4
br label %denominator.loop
denominator.loop:
%denominator.key = phi i32 [ %lid, %maximum.group.skip ], [ %denominator.key.next, %denominator.step ]
%denominator.value = phi float [ 0x0000000000000000, %maximum.group.skip ], [ %denominator.next, %denominator.step ]
%denominator.more = icmp ult i32 %denominator.key, %active.limit
br i1 %denominator.more, label %denominator.step, label %denominator.wave.loop
denominator.step:
%denominator.row = mul i32 %group, %length
%denominator.slot = add i32 %denominator.row, %denominator.key
%denominator.slot.wide = zext i32 %denominator.slot to i64
%denominator.score = call float @attention_step_score_load(ptr addrspace(1) %context, i64 %denominator.slot.wide)
%denominator.centered = call float @recipe.state.sub(float %denominator.score, float %maximum.global)
%denominator.exp = call float @recipe.state.exp(float %denominator.centered)
%denominator.next = call float @recipe.state.add(float %denominator.value, float %denominator.exp)
%denominator.key.next = add i32 %denominator.key, %block
br label %denominator.loop
denominator.wave.loop:
%denominator.offset = phi i32 [ %wave.half.start, %denominator.loop ], [ %denominator.offset.next, %denominator.wave.step ]
%denominator.wave.value = phi float [ %denominator.value, %denominator.loop ], [ %denominator.wave.value.next, %denominator.wave.step ]
%denominator.wave.more = icmp ne i32 %denominator.offset, 0
br i1 %denominator.wave.more, label %denominator.wave.step, label %denominator.wave.done
denominator.wave.step:
%denominator.partner.lane = xor i32 %lane, %denominator.offset
%denominator.partner.index = mul i32 %denominator.partner.lane, 4
%denominator.partner = call float @recipe.wave.partner(float %denominator.wave.value, i32 %denominator.partner.index)
%denominator.wave.value.next = call float @recipe.state.add(float %denominator.wave.value, float %denominator.partner)
%denominator.offset.next = lshr i32 %denominator.offset, 1
br label %denominator.wave.loop
denominator.wave.done:
%denominator.owner = icmp eq i32 %lane, 0
br i1 %denominator.owner, label %denominator.wave.store, label %denominator.wave.skip
denominator.wave.store:
%denominator.wave.ptr = getelementptr float, ptr addrspace(3) %tile.float, i32 %wave
store float %denominator.wave.value, ptr addrspace(3) %denominator.wave.ptr, align 4
br label %denominator.wave.skip
denominator.wave.skip:
call void @llvm.nvvm.barrier0()
%denominator.group.owner = icmp eq i32 %lid, 0
br i1 %denominator.group.owner, label %denominator.group.loop, label %denominator.group.skip
denominator.group.loop:
%denominator.wave.index = phi i32 [ 0, %denominator.wave.skip ], [ %denominator.wave.index.next, %denominator.group.step ]
%denominator.group.value = phi float [ 0x0000000000000000, %denominator.wave.skip ], [ %denominator.group.next, %denominator.group.step ]
%denominator.group.more = icmp ult i32 %denominator.wave.index, %waves
br i1 %denominator.group.more, label %denominator.group.step, label %denominator.group.done
denominator.group.step:
%denominator.group.ptr = getelementptr float, ptr addrspace(3) %tile.float, i32 %denominator.wave.index
%denominator.group.wave = load float, ptr addrspace(3) %denominator.group.ptr, align 4
%denominator.group.next = call float @recipe.state.add(float %denominator.group.value, float %denominator.group.wave)
%denominator.wave.index.next = add i32 %denominator.wave.index, 1
br label %denominator.group.loop
denominator.group.done:
store float %denominator.group.value, ptr addrspace(3) %denominator.global.ptr, align 4
br label %denominator.group.skip
denominator.group.skip:
call void @llvm.nvvm.barrier0()
%denominator.global = load float, ptr addrspace(3) %denominator.global.ptr, align 4
br label %probability.loop
probability.loop:
%probability.key = phi i32 [ %lid, %denominator.group.skip ], [ %probability.key.next, %probability.step ]
%probability.more = icmp ult i32 %probability.key, %active.limit
br i1 %probability.more, label %probability.step, label %probability.done
probability.step:
%probability.row = mul i32 %group, %length
%probability.slot = add i32 %probability.row, %probability.key
%probability.slot.wide = zext i32 %probability.slot to i64
%probability.score = call float @attention_step_score_load(ptr addrspace(1) %context, i64 %probability.slot.wide)
%probability.centered = call float @recipe.state.sub(float %probability.score, float %maximum.global)
%probability.exp = call float @recipe.state.exp(float %probability.centered)
%probability.value = call float @recipe.state.div(float %probability.exp, float %denominator.global)
call void @attention_step_score_store(ptr addrspace(1) %context, i64 %probability.slot.wide, float %probability.value)
%probability.key.next = add i32 %probability.key, %block
br label %probability.loop
probability.done:
call void @grid_barrier(i32 %threads)
br label %output.channel.loop
output.channel.loop:
%output.channel = phi i32 [ %global.wave, %probability.done ], [ %output.channel.next, %output.channel.done ]
%output.channel.more = icmp ult i32 %output.channel, %channels
br i1 %output.channel.more, label %output.key.loop, label %output.stats.owner
output.key.loop:
%output.key = phi i32 [ %lane, %output.channel.loop ], [ %output.key.next, %output.key.step ]
%output.sum.0 = phi float [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.0.next, %output.key.step ]
%output.sum.1 = phi float [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.1.next, %output.key.step ]
%output.sum.2 = phi float [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.2.next, %output.key.step ]
%output.sum.3 = phi float [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.3.next, %output.key.step ]
%output.key.more = icmp ult i32 %output.key, %reached
br i1 %output.key.more, label %output.key.step, label %output.wave.prepare
output.key.step:
%output.head = udiv i32 %output.channel, %width
%output.row = mul i32 %output.head, %length
%output.kv.head = udiv i32 %output.head, %kv.group
%output.kv.channel.base = mul i32 %output.kv.head, %width
%output.local.channel = urem i32 %output.channel, %width
%output.kv.channel = add i32 %output.kv.channel.base, %output.local.channel
%output.kv.channel.offset = mul i32 %output.kv.channel, %length
%output.key.0 = add i32 %output.key, 0
%output.key.1 = add i32 %output.key, %wave.width
%output.key.2 = add i32 %output.key.1, %wave.width
%output.key.3 = add i32 %output.key.2, %wave.width
%output.active.0 = icmp ult i32 %output.key.0, %reached
%output.active.1 = icmp ult i32 %output.key.1, %reached
%output.active.2 = icmp ult i32 %output.key.2, %reached
%output.active.3 = icmp ult i32 %output.key.3, %reached
%output.safe.0 = select i1 %output.active.0, i32 %output.key.0, i32 0
%output.safe.1 = select i1 %output.active.1, i32 %output.key.1, i32 0
%output.safe.2 = select i1 %output.active.2, i32 %output.key.2, i32 0
%output.safe.3 = select i1 %output.active.3, i32 %output.key.3, i32 0
%output.slot.0 = add i32 %output.row, %output.safe.0
%output.slot.1 = add i32 %output.row, %output.safe.1
%output.slot.2 = add i32 %output.row, %output.safe.2
%output.slot.3 = add i32 %output.row, %output.safe.3
%output.slot.0.wide = zext i32 %output.slot.0 to i64
%output.slot.1.wide = zext i32 %output.slot.1 to i64
%output.slot.2.wide = zext i32 %output.slot.2 to i64
%output.slot.3.wide = zext i32 %output.slot.3 to i64
%output.probability.0.raw = call float @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.0.wide)
%output.probability.1.raw = call float @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.1.wide)
%output.probability.2.raw = call float @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.2.wide)
%output.probability.3.raw = call float @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.3.wide)
%output.probability.0 = select i1 %output.active.0, float %output.probability.0.raw, float 0x0000000000000000
%output.probability.1 = select i1 %output.active.1, float %output.probability.1.raw, float 0x0000000000000000
%output.probability.2 = select i1 %output.active.2, float %output.probability.2.raw, float 0x0000000000000000
%output.probability.3 = select i1 %output.active.3, float %output.probability.3.raw, float 0x0000000000000000
%output.index.0.local = add i32 %output.kv.channel.offset, %output.safe.0
%output.index.1.local = add i32 %output.kv.channel.offset, %output.safe.1
%output.index.2.local = add i32 %output.kv.channel.offset, %output.safe.2
%output.index.3.local = add i32 %output.kv.channel.offset, %output.safe.3
%output.index.0 = add i32 %kv.plane, %output.index.0.local
%output.index.1 = add i32 %kv.plane, %output.index.1.local
%output.index.2 = add i32 %kv.plane, %output.index.2.local
%output.index.3 = add i32 %kv.plane, %output.index.3.local
%output.index.0.wide = zext i32 %output.index.0 to i64
%output.index.1.wide = zext i32 %output.index.1 to i64
%output.index.2.wide = zext i32 %output.index.2 to i64
%output.index.3.wide = zext i32 %output.index.3 to i64
%output.ptr.0 = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %output.index.0.wide
%output.ptr.1 = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %output.index.1.wide
%output.ptr.2 = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %output.index.2.wide
%output.ptr.3 = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %output.index.3.wide
%output.value.0.raw = load i8, ptr addrspace(1) %output.ptr.0, align 1
%output.value.1.raw = load i8, ptr addrspace(1) %output.ptr.1, align 1
%output.value.2.raw = load i8, ptr addrspace(1) %output.ptr.2, align 1
%output.value.3.raw = load i8, ptr addrspace(1) %output.ptr.3, align 1
%output.value.0.float = call float @recipe.kv.to.state(i8 %output.value.0.raw)
%output.value.1.float = call float @recipe.kv.to.state(i8 %output.value.1.raw)
%output.value.2.float = call float @recipe.kv.to.state(i8 %output.value.2.raw)
%output.value.3.float = call float @recipe.kv.to.state(i8 %output.value.3.raw)
%output.value.0 = select i1 %output.active.0, float %output.value.0.float, float 0x0000000000000000
%output.value.1 = select i1 %output.active.1, float %output.value.1.float, float 0x0000000000000000
%output.value.2 = select i1 %output.active.2, float %output.value.2.float, float 0x0000000000000000
%output.value.3 = select i1 %output.active.3, float %output.value.3.float, float 0x0000000000000000
%output.sum.0.next = call float @recipe.state.madd(float %output.sum.0, float %output.probability.0, float %output.value.0)
%output.sum.1.next = call float @recipe.state.madd(float %output.sum.1, float %output.probability.1, float %output.value.1)
%output.sum.2.next = call float @recipe.state.madd(float %output.sum.2, float %output.probability.2, float %output.value.2)
%output.sum.3.next = call float @recipe.state.madd(float %output.sum.3, float %output.probability.3, float %output.value.3)
%output.key.stride = mul i32 %wave.width, 4
%output.key.next = add i32 %output.key, %output.key.stride
br label %output.key.loop
output.wave.prepare:
%output.sum.01 = call float @recipe.state.add(float %output.sum.0, float %output.sum.1)
%output.sum.23 = call float @recipe.state.add(float %output.sum.2, float %output.sum.3)
%output.sum = call float @recipe.state.add(float %output.sum.01, float %output.sum.23)
br label %output.wave.loop
output.wave.loop:
%output.offset = phi i32 [ %wave.half.start, %output.wave.prepare ], [ %output.offset.next, %output.wave.step ]
%output.wave.sum = phi float [ %output.sum, %output.wave.prepare ], [ %output.wave.sum.next, %output.wave.step ]
%output.wave.more = icmp ne i32 %output.offset, 0
br i1 %output.wave.more, label %output.wave.step, label %output.wave.done
output.wave.step:
%output.partner.lane = xor i32 %lane, %output.offset
%output.partner.index = mul i32 %output.partner.lane, 4
%output.partner = call float @recipe.wave.partner(float %output.wave.sum, i32 %output.partner.index)
%output.wave.sum.next = call float @recipe.state.add(float %output.wave.sum, float %output.partner)
%output.offset.next = lshr i32 %output.offset, 1
br label %output.wave.loop
output.wave.done:
%output.owner = icmp eq i32 %lane, 0
br i1 %output.owner, label %output.store, label %output.channel.done
output.store:
%output.position.base = mul i32 %output.channel, %length
%output.position.index = add i32 %output.position.base, %position
%output.position.wide = zext i32 %output.position.index to i64
%output.position.phys = call i64 @recipe.window.index(i64 %output.position.wide, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.position.phys
%output.half.value = call i8 @recipe.model.from.state(float %output.wave.sum)
store i8 %output.half.value, ptr addrspace(1) %output.ptr, align 1
br label %output.channel.done
output.channel.done:
%output.channel.next = add i32 %output.channel, %global.waves
br label %output.channel.loop
output.stats.owner:
br label %exit
exit:
ret void
}
; True when the query keeps the block that holds this key. Each query owns one
; row of block scores followed by one admission flag per block.
define internal i1 @attention_selected(ptr addrspace(1) nocapture readonly %context, i64 %score.row, i32 %blocks, i32 %select.block, i32 %query, i32 %key) #1 { entry:
%blocks.wide = zext i32 %blocks to i64 %stride.wide = mul i64 %blocks.wide, 2
%query.wide = zext i32 %query to i64
%row = mul i64 %query.wide, %stride.wide
%start = add i64 %score.row, %row
%block.index = udiv i32 %key, %select.block
%flag.local = add i32 %blocks, %block.index
%flag.local.wide = zext i32 %flag.local to i64
%flag.index = add i64 %start, %flag.local.wide
%flag.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %flag.index
%flag = load i8, ptr addrspace(1) %flag.ptr, align 1
%result = call i1 @recipe.ogt(i8 %flag, i8 48)
ret i1 %result
}
; Append one causal block representative. Running sums round after each key;
; mean pooling, trained normalization, and rotary each execute once per prefix.
define internal void @attention_index_cache(ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) %context,
i64 %sum.start, i64 %cache.start, i32 %filled, i32 %block.index, i32 %select.block, i32 %index.heads, i32 %index.width,
i32 %mode, i32 %dims, float %base, i8 %epsilon, i1 %pooled) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false)
%filled.value = call i8 @recipe.from.u32(i32 %filled)
br label %mean.loop
mean.loop:
%d = phi i32 [ 0, %entry ], [ %d.next, %mean.next ]
%sum = phi float [ %state.zero, %entry ], [ %sum.next, %mean.next ]
%more = icmp ult i32 %d, %index.width
br i1 %more, label %mean.step, label %norm.prepare
mean.step:
%d.wide = zext i32 %d to i64
%sum.index = add i64 %sum.start, %d.wide
%sum.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %sum.index
%total = load i8, ptr addrspace(1) %sum.ptr, align 1
%mean = call i8 @recipe.div(i8 %total, i8 %filled.value)
%mean.index = add i64 %cache.start, %d.wide
%mean.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %mean.index
store i8 %mean, ptr addrspace(1) %mean.ptr, align 1
br i1 %pooled, label %mean.square, label %mean.next
mean.square:
%mean.wide = call float @recipe.decode(i8 %mean)
%square = call float @recipe.state.mul(float %mean.wide, float %mean.wide)
%sum.added = call float @recipe.state.add(float %sum, float %square)
br label %mean.next
mean.next:
%sum.next = phi float [ %sum, %mean.step ], [ %sum.added, %mean.square ]
%d.next = add i32 %d, 1
br label %mean.loop
norm.prepare:
br i1 %pooled, label %norm.mode, label %rope.prepare
norm.mode:
%rms = icmp eq i32 %mode, 2
%epsilon.wide = call float @recipe.decode(i8 %epsilon)
br i1 %rms, label %norm.rms, label %norm.l2
norm.rms:
%width.value = call float @recipe.state.from.u32(i32 %index.width)
%variance = call float @recipe.state.div(float %sum, float %width.value)
%adjusted = call float @recipe.state.add(float %variance, float %epsilon.wide)
%rms.deviation = call float @recipe.state.sqrt(float %adjusted)
br label %norm.loop.prepare
norm.l2:
%root = call float @recipe.state.sqrt(float %sum)
%root.model = call i8 @recipe.encode(float %root)
%epsilon.model = call i8 @recipe.encode(float %epsilon.wide)
%above = call i1 @recipe.ogt(i8 %root.model, i8 %epsilon.model)
%l2.deviation = select i1 %above, float %root, float %epsilon.wide
br label %norm.loop.prepare
norm.loop.prepare:
%deviation = phi float [ %rms.deviation, %norm.rms ], [ %l2.deviation, %norm.l2 ]
%heads.wide = zext i32 %index.heads to i64
%width.wide = zext i32 %index.width to i64
%scale.base = mul i64 %heads.wide, %width.wide
br label %norm.loop
norm.loop:
%norm.d = phi i32 [ 0, %norm.loop.prepare ], [ %norm.next, %norm.store ]
%norm.more = icmp ult i32 %norm.d, %index.width
br i1 %norm.more, label %norm.step, label %rope.prepare
norm.step:
%norm.d.wide = zext i32 %norm.d to i64
%norm.index = add i64 %cache.start, %norm.d.wide
%norm.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %norm.index
%norm.mean = load i8, ptr addrspace(1) %norm.ptr, align 1
%norm.mean.wide = call float @recipe.decode(i8 %norm.mean)
%unit = call float @recipe.state.div(float %norm.mean.wide, float %deviation)
br i1 %rms, label %norm.scale, label %norm.plain
norm.scale:
%scale.index = add i64 %scale.base, %norm.d.wide
%scale.ptr = getelementptr inbounds i8, ptr addrspace(1) %key.weights, i64 %scale.index
%scale.model = load i8, ptr addrspace(1) %scale.ptr, align 1
%scale = call float @recipe.decode(i8 %scale.model)
%scaled = call float @recipe.state.mul(float %unit, float %scale)
br label %norm.store
norm.plain:
br label %norm.store
norm.store:
%norm.value = phi float [ %scaled, %norm.scale ], [ %unit, %norm.plain ]
%norm.model = call i8 @recipe.encode(float %norm.value)
store i8 %norm.model, ptr addrspace(1) %norm.ptr, align 1
%norm.next = add i32 %norm.d, 1
br label %norm.loop
rope.prepare:
%half = udiv i32 %dims, 2
%rotated = icmp ne i32 %half, 0
br i1 %rotated, label %rope.constants, label %exit
rope.constants:
%log.base = call float @recipe.state.log(float %base)
%dims.value = call float @recipe.state.from.u32(i32 %dims)
%position = mul i32 %block.index, %select.block
%position.value = call float @recipe.state.from.u32(i32 %position)
br label %rope.loop
rope.loop:
%rope.d = phi i32 [ 0, %rope.constants ], [ %rope.next, %rope.step ]
%rope.more = icmp ult i32 %rope.d, %half
br i1 %rope.more, label %rope.step, label %exit
rope.step:
%partner = add i32 %rope.d, %half
%rope.d.wide = zext i32 %rope.d to i64
%partner.wide = zext i32 %partner to i64
%lower.index = add i64 %cache.start, %rope.d.wide
%upper.index = add i64 %cache.start, %partner.wide
%lower.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %lower.index
%upper.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %upper.index
%lower.model = load i8, ptr addrspace(1) %lower.ptr, align 1
%upper.model = load i8, ptr addrspace(1) %upper.ptr, align 1
%lower = call float @recipe.decode(i8 %lower.model)
%upper = call float @recipe.decode(i8 %upper.model)
%two.index = mul i32 %rope.d, 2
%two.value = call float @recipe.state.from.u32(i32 %two.index)
%ratio = call float @recipe.state.div(float %two.value, float %dims.value)
%exponent.positive = call float @recipe.state.mul(float %ratio, float %log.base)
%exponent = call float @recipe.state.neg(float %exponent.positive)
%frequency = call float @recipe.state.exp(float %exponent)
%angle = call float @recipe.state.mul(float %position.value, float %frequency)
%cos = call float @recipe.state.cos(float %angle)
%sin = call float @recipe.state.sin(float %angle)
%lower.cos = call float @recipe.state.mul(float %lower, float %cos)
%upper.sin = call float @recipe.state.mul(float %upper, float %sin)
%upper.cos = call float @recipe.state.mul(float %upper, float %cos)
%lower.sin = call float @recipe.state.mul(float %lower, float %sin)
%lower.result = call float @recipe.state.sub(float %lower.cos, float %upper.sin)
%upper.result = call float @recipe.state.add(float %upper.cos, float %lower.sin)
%lower.out = call i8 @recipe.encode(float %lower.result)
%upper.out = call i8 @recipe.encode(float %upper.result)
store i8 %lower.out, ptr addrspace(1) %lower.ptr, align 1
store i8 %upper.out, ptr addrspace(1) %upper.ptr, align 1
%rope.next = add i32 %rope.d, 1
br label %rope.loop
exit:
ret void
}

; The running sum of indexer keys of one key block, extended by the keys of the
; block that lie in the forward window `begin..end`. A block whose first key
; lies in the window starts from zero; one that began earlier keeps the sum an
; earlier window left, so a decode step adds one key to one block.
define internal void @attention_index_body( ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) %context,
i64 %p, i32 %begin, i32 %end, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width,
i32 %select.block, i1 %gate, i8 %epsilon, i32 %index.mode, i32 %index.dims, i1 %index.pooled, float %index.base, i32 %input.length, i32 %input.origin ) #1 { entry:
%from.wide = zext i32 %from to i64 %channels.wide = zext i32 %channels to i64 %rows.wide = zext i32 %rows to i64 %heads.wide = zext i32 %heads to i64 %index.heads.wide = zext i32 %index.heads to i64 %index.width.wide = zext i32 %index.width to i64 %select.block.wide = zext i32 %select.block to i64 %begin.wide = zext i32 %begin to i64 %end.wide = zext i32 %end to i64
%length = udiv i64 %from.wide, %channels.wide
%input.length.wide = zext i32 %input.length to i64 %input.origin.wide = zext i32 %input.origin to i64
%index.query.channels = mul i64 %index.heads.wide, %index.width.wide
%index.channels = add i64 %index.query.channels, %index.width.wide
%row.stride = mul i64 %index.channels, %input.length.wide
%index.key.base = mul i64 %index.query.channels, %input.length.wide
%blocks.numerator = add i64 %length, %select.block.wide
%blocks.less = sub i64 %blocks.numerator, 1
%blocks = udiv i64 %blocks.less, %select.block.wide
%statistics.rows = mul i64 %rows.wide, %heads.wide
%statistics.plane = mul i64 %statistics.rows, %length
%representative.base = mul i64 %statistics.plane, 2
%representative.stride = mul i64 %blocks, %index.width.wide
%representative.total = mul i64 %representative.stride, %rows.wide
%score.base = add i64 %representative.base, %representative.total
%queries = mul i64 %rows.wide, %length
%score.stride = mul i64 %blocks, 2
%score.cache.total = mul i64 %queries, %score.stride
%raw.base = add i64 %score.base, %score.cache.total
%cache.total = mul i64 %queries, %index.width.wide
%cache.base = add i64 %raw.base, %cache.total
%prefix.base = add i64 %cache.base, %cache.total
%counts.base = add i64 %prefix.base, %cache.total
%counts.byte.base = mul i64 %counts.base, 1
%counts.ptr = getelementptr i8, ptr addrspace(1) %context, i64 %counts.byte.base
%count.ptr = getelementptr i64, ptr addrspace(1) %counts.ptr, i64 %p
%row = udiv i64 %p, %blocks
%block.index = urem i64 %p, %blocks
%row.base = mul i64 %row, %row.stride
%key.origin = add i64 %row.base, %index.key.base
%cache.row.positions = mul i64 %row, %length
%cache.row = mul i64 %cache.row.positions, %index.width.wide
%representative.row = mul i64 %row, %representative.stride
%representative.block = mul i64 %block.index, %index.width.wide
%representative.start.row = add i64 %representative.base, %representative.row
%representative.start = add i64 %representative.start.row, %representative.block
%start = mul i64 %block.index, %select.block.wide
%stop.full = add i64 %start, %select.block.wide
%stop.over = icmp ugt i64 %stop.full, %end.wide
%stop = select i1 %stop.over, i64 %end.wide, i64 %stop.full
%first.before = icmp ult i64 %start, %begin.wide
%first = select i1 %first.before, i64 %begin.wide, i64 %start
%fresh = icmp uge i64 %start, %begin.wide
%extends = icmp ult i64 %first, %stop
%clear = and i1 %fresh, %extends
%state.zero = call float @recipe.state.from.u1(i1 false)
%model.zero = call i8 @recipe.encode(float %state.zero)
%prior.count = load i64, ptr addrspace(1) %count.ptr, align 1
%initial.count = select i1 %clear, i64 0, i64 %prior.count
br i1 %clear, label %clear.loop, label %key.loop
clear.loop:
%clear.d = phi i32 [ 0, %entry ], [ %clear.next, %clear.step ]
%clear.more = icmp ult i32 %clear.d, %index.width
br i1 %clear.more, label %clear.step, label %key.loop
clear.step:
%clear.d.wide = zext i32 %clear.d to i64
%clear.index = add i64 %representative.start, %clear.d.wide
%clear.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %clear.index
store i8 %model.zero, ptr addrspace(1) %clear.ptr, align 1
%clear.next = add i32 %clear.d, 1
br label %clear.loop
key.loop:
%key = phi i64 [ %first, %entry ], [ %first, %clear.loop ], [ %key.advance, %key.step ]
%filled = phi i64 [ %initial.count, %entry ], [ %initial.count, %clear.loop ], [ %filled.next, %key.step ]
%key.more = icmp ult i64 %key, %stop
br i1 %key.more, label %key.prepare, label %exit
key.prepare:
%key.relative = sub i64 %key, %input.origin.wide
%key.position = add i64 %key.origin, %key.relative
%cache.position = mul i64 %key, %index.width.wide
%cache.local = add i64 %cache.row, %cache.position
%raw.start = add i64 %raw.base, %cache.local
%cache.start = add i64 %cache.base, %cache.local
%prefix.start = add i64 %prefix.base, %cache.local
br label %dim.loop
dim.loop:
%dim = phi i32 [ 0, %key.prepare ], [ %dim.advance, %dim.step ]
%dim.more = icmp ult i32 %dim, %index.width
br i1 %dim.more, label %dim.step, label %key.step
dim.step:
%dim.wide = zext i32 %dim to i64 %dim.offset = mul i64 %dim.wide, %input.length.wide
%dim.index = add i64 %key.position, %dim.offset
%dim.ptr = getelementptr inbounds i8, ptr addrspace(1) %indexer, i64 %dim.index
%dim.value = load i8, ptr addrspace(1) %dim.ptr, align 1
%raw.index = add i64 %raw.start, %dim.wide
%raw.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %raw.index
store i8 %dim.value, ptr addrspace(1) %raw.ptr, align 1
%dim.target = add i64 %representative.start, %dim.wide
%dim.target.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %dim.target
%dim.prior = load i8, ptr addrspace(1) %dim.target.ptr, align 1
%dim.sum = call i8 @recipe.add(i8 %dim.prior, i8 %dim.value)
store i8 %dim.sum, ptr addrspace(1) %dim.target.ptr, align 1
%prefix.index = add i64 %prefix.start, %dim.wide
%prefix.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %prefix.index
store i8 %dim.sum, ptr addrspace(1) %prefix.ptr, align 1
%dim.advance = add i32 %dim, 1
br label %dim.loop
key.step:
%filled.next = add i64 %filled, 1
%filled.i32 = trunc i64 %filled.next to i32
%block.i32 = trunc i64 %block.index to i32
call void @attention_index_cache(ptr addrspace(1) %key.weights, ptr addrspace(1) %context,
i64 %representative.start, i64 %cache.start, i32 %filled.i32, i32 %block.i32, i32 %select.block, i32 %index.heads, i32 %index.width,
i32 %index.mode, i32 %index.dims, float %index.base, i8 %epsilon, i1 %index.pooled)
store i64 %filled.next, ptr addrspace(1) %count.ptr, align 1
%key.advance = add i64 %key, 1
br label %key.loop
exit:
ret void
}
; Each query scores cached causal representatives. Its own block uses that
; query's prefix; an earlier block uses its final filled position.
define internal void @attention_select_body( ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) %context,
i64 %p, i32 %keep, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads, i32 %value.heads, i32 %index.heads,
i32 %index.width, i32 %select.block, i1 %gate, i8 %epsilon, i32 %index.mode, i32 %index.dims, i1 %index.pooled, float %index.base, i32 %input.length, i32 %input.origin ) #1 { entry:
%from.wide = zext i32 %from to i64 %channels.wide = zext i32 %channels to i64 %rows.wide = zext i32 %rows to i64 %heads.wide = zext i32 %heads to i64 %index.heads.wide = zext i32 %index.heads to i64 %index.width.wide = zext i32 %index.width to i64 %select.block.wide = zext i32 %select.block to i64
%length = udiv i64 %from.wide, %channels.wide
%input.length.wide = zext i32 %input.length to i64 %input.origin.wide = zext i32 %input.origin to i64
%index.query.channels = mul i64 %index.heads.wide, %index.width.wide
%index.channels = add i64 %index.query.channels, %index.width.wide
%row.stride = mul i64 %index.channels, %input.length.wide
%index.key.base = mul i64 %index.query.channels, %input.length.wide
%blocks.numerator = add i64 %length, %select.block.wide
%blocks.less = sub i64 %blocks.numerator, 1
%blocks = udiv i64 %blocks.less, %select.block.wide
%score.stride = mul i64 %blocks, 2
%statistics.rows = mul i64 %rows.wide, %heads.wide
%statistics.plane = mul i64 %statistics.rows, %length
%representative.base = mul i64 %statistics.plane, 2
%representative.stride = mul i64 %blocks, %index.width.wide
%representative.total = mul i64 %representative.stride, %rows.wide
%score.base = add i64 %representative.base, %representative.total
%row = udiv i64 %p, %length
%query = urem i64 %p, %length
%queries = mul i64 %rows.wide, %length
%score.cache.total = mul i64 %queries, %score.stride
%raw.base = add i64 %score.base, %score.cache.total
%cache.total = mul i64 %queries, %index.width.wide
%cache.base = add i64 %raw.base, %cache.total
%cache.row.positions = mul i64 %row, %length
%cache.row = mul i64 %cache.row.positions, %index.width.wide
%row.base = mul i64 %row, %row.stride
%key.origin = add i64 %row.base, %index.key.base
%query.relative = sub i64 %query, %input.origin.wide
%query.position = add i64 %row.base, %query.relative
%count.less = udiv i64 %query, %select.block.wide
%count.wide = add i64 %count.less, 1
%count = trunc i64 %count.wide to i32
%score.query = mul i64 %p, %score.stride
%score.start = add i64 %score.base, %score.query
%representative.row = mul i64 %row, %representative.stride
%representative.start = add i64 %representative.base, %representative.row
%query.i32 = trunc i64 %query to i32 %length.i32 = trunc i64 %length to i32
%state.zero = call float @recipe.state.from.u1(i1 false)
%model.zero = call i8 @recipe.encode(float %state.zero)
br label %clear.loop
clear.loop:
%clear.b = phi i32 [ 0, %entry ], [ %clear.next, %clear.step ]
%clear.more = icmp ult i32 %clear.b, %count
br i1 %clear.more, label %clear.step, label %head.loop
clear.step:
%clear.b.wide = zext i32 %clear.b to i64 %clear.index = add i64 %score.start, %clear.b.wide
%clear.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %clear.index
store i8 %model.zero, ptr addrspace(1) %clear.ptr, align 1
%clear.next = add i32 %clear.b, 1
br label %clear.loop
head.loop:
%head = phi i32 [ 0, %clear.loop ], [ %head.advance, %head.done ]
%head.more = icmp ult i32 %head, %index.heads
br i1 %head.more, label %head.prepare, label %threshold.prepare
head.prepare:
%head.wide = zext i32 %head to i64 %head.offset = mul i64 %head.wide, %index.width.wide
%head.plane = mul i64 %head.offset, %input.length.wide
%head.base = add i64 %query.position, %head.plane
br label %score.loop
score.loop:
%score.b = phi i32 [ 0, %head.prepare ], [ %score.advance, %score.store ]
%score.more = icmp ult i32 %score.b, %count
br i1 %score.more, label %score.prepare, label %head.done
score.prepare:
%score.block.wide = zext i32 %score.b to i64
%score.block.next = add i64 %score.block.wide, 1
%score.block.end = mul i64 %score.block.next, %select.block.wide
%score.block.last = sub i64 %score.block.end, 1
%score.block.future = icmp ugt i64 %score.block.last, %query
%score.position = select i1 %score.block.future, i64 %query, i64 %score.block.last
%score.cache.position = mul i64 %score.position, %index.width.wide
%score.cache.local = add i64 %cache.row, %score.cache.position
%score.cache.start = add i64 %cache.base, %score.cache.local
br label %score.dim.loop
score.dim.loop:
%score.d = phi i32 [ 0, %score.prepare ], [ %score.d.advance, %score.dim.step ]
%score.sum = phi float [ %state.zero, %score.prepare ], [ %score.sum.next, %score.dim.step ]
%score.dim.more = icmp ult i32 %score.d, %index.width
br i1 %score.dim.more, label %score.dim.step, label %score.store
score.dim.step:
%score.d.wide = zext i32 %score.d to i64 %score.dim.offset = mul i64 %score.d.wide, %input.length.wide
%score.query.index = add i64 %head.base, %score.dim.offset
%score.query.ptr = getelementptr inbounds i8, ptr addrspace(1) %indexer, i64 %score.query.index
%score.query.value = load i8, ptr addrspace(1) %score.query.ptr, align 1
%score.cache.index = add i64 %score.cache.start, %score.d.wide
%score.cache.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %score.cache.index
%score.representative.value = load i8, ptr addrspace(1) %score.cache.ptr, align 1
%score.query.wide = call float @recipe.decode(i8 %score.query.value)
%score.representative.wide = call float @recipe.decode(i8 %score.representative.value)
%score.term = call float @recipe.state.mul(float %score.query.wide, float %score.representative.wide)
%score.sum.next = call float @recipe.state.add(float %score.sum, float %score.term)
%score.d.advance = add i32 %score.d, 1
br label %score.dim.loop
score.store:
%score.sum.model = call i8 @recipe.encode(float %score.sum)
%score.positive = call i1 @recipe.ogt(i8 %score.sum.model, i8 0)
%score.head.model = select i1 %score.positive, i8 %score.sum.model, i8 %model.zero
%score.head.wide = call float @recipe.decode(i8 %score.head.model)
%score.b.wide = zext i32 %score.b to i64 %score.index = add i64 %score.start, %score.b.wide
%score.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %score.index
%score.prior = load i8, ptr addrspace(1) %score.ptr, align 1
%score.prior.wide = call float @recipe.decode(i8 %score.prior)
%score.total = call float @recipe.state.add(float %score.prior.wide, float %score.head.wide)
%score.value = call i8 @recipe.encode(float %score.total)
store i8 %score.value, ptr addrspace(1) %score.ptr, align 1
%score.advance = add i32 %score.b, 1
br label %score.loop
head.done:
%head.advance = add i32 %head, 1
br label %head.loop
threshold.prepare:
%flag.base = add i64 %score.start, %blocks
br label %rank.loop
rank.loop:
%rank.b = phi i32 [ 0, %threshold.prepare ], [ %rank.b.next, %rank.store ]
%rank.more = icmp ult i32 %rank.b, %count
br i1 %rank.more, label %rank.body, label %select.exit
rank.body:
%rank.b.wide = zext i32 %rank.b to i64 %rank.b.index = add i64 %score.start, %rank.b.wide
%rank.b.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %rank.b.index
%rank.b.score = load i8, ptr addrspace(1) %rank.b.ptr, align 1
br label %rank.inner
rank.inner:
%rank.c = phi i32 [ 0, %rank.body ], [ %rank.c.next, %rank.inner.step ]
%rank.ahead = phi i32 [ 0, %rank.body ], [ %rank.ahead.next, %rank.inner.step ]
%rank.inner.more = icmp ult i32 %rank.c, %count
br i1 %rank.inner.more, label %rank.inner.step, label %rank.decide
rank.inner.step:
%rank.c.wide = zext i32 %rank.c to i64 %rank.c.index = add i64 %score.start, %rank.c.wide
%rank.c.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %rank.c.index
%rank.c.score = load i8, ptr addrspace(1) %rank.c.ptr, align 1
%rank.greater = call i1 @recipe.ogt(i8 %rank.c.score, i8 %rank.b.score)
%rank.same = call i1 @recipe.oeq(i8 %rank.c.score, i8 %rank.b.score)
%rank.earlier = icmp ult i32 %rank.c, %rank.b
%rank.tie = and i1 %rank.same, %rank.earlier
%rank.before = or i1 %rank.greater, %rank.tie
%rank.one = zext i1 %rank.before to i32
%rank.ahead.next = add i32 %rank.ahead, %rank.one
%rank.c.next = add i32 %rank.c, 1
br label %rank.inner
rank.decide:
%rank.admit = icmp ult i32 %rank.ahead, %keep
br label %rank.store
rank.store:
%rank.flag = call i8 @recipe.from.u1(i1 %rank.admit)
%rank.store.b.wide = zext i32 %rank.b to i64 %rank.flag.index = add i64 %flag.base, %rank.store.b.wide
%rank.flag.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %rank.flag.index
store i8 %rank.flag, ptr addrspace(1) %rank.flag.ptr, align 1
%rank.b.next = add i32 %rank.b, 1
br label %rank.loop
select.exit:
ret void
}

define internal void @attention_tile_products(ptr addrspace(1) nocapture readonly %output, i64 %output.row,
i32 %delta.base, i32 %product.base, i32 %query.base, i32 %query.count, i32 %head.start,
i32 %head.width, i32 %length, i32 %lid, i32 %block) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%output.row.wide = add i64 %output.row, 0 %delta.base.wide = zext i32 %delta.base to i64 %product.base.wide = zext i32 %product.base to i64 %query.base.wide = zext i32 %query.base to i64 %head.start.wide = zext i32 %head.start to i64 %head.width.wide = zext i32 %head.width to i64 %length.wide = zext i32 %length to i64
%zero = call float @recipe.state.from.u1(i1 false)
br label %query.loop
query.loop:
%query = phi i32 [ %lid, %entry ], [ %query.next, %store ]
%query.wide = zext i32 %query to i64
%query.more = icmp ult i32 %query, %query.count
br i1 %query.more, label %prepare, label %exit
prepare:
br label %channel.loop
channel.loop:
%channel = phi i32 [ 0, %prepare ], [ %channel.next, %channel.step ]
%sum = phi float [ %zero, %prepare ], [ %sum.next, %channel.step ]
%channel.more = icmp ult i32 %channel, %head.width
br i1 %channel.more, label %channel.step, label %store
channel.step:
%channel.wide = zext i32 %channel to i64 %shared.row = mul i64 %query.wide, %head.width.wide
%shared.local = add i64 %shared.row, %channel.wide
%delta.index = add i64 %delta.base.wide, %shared.local
%delta.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i64 %delta.index
%delta = load float, ptr addrspace(3) %delta.ptr, align 4
%output.channel = add i64 %head.start.wide, %channel.wide
%output.channel.base = mul i64 %output.channel, %length.wide
%position = add i64 %query.base.wide, %query.wide
%output.local = add i64 %output.channel.base, %position
%output.index = add i64 %output.row.wide, %output.local
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.index
%output.value.model = load i8, ptr addrspace(1) %output.ptr, align 1
%output.value = call float @recipe.decode(i8 %output.value.model)
%delta.wide = select i1 true, float %delta, float %delta
%output.wide = select i1 true, float %output.value, float %output.value
%term = call float @recipe.state.mul(float %delta.wide, float %output.wide)
%sum.next = call float @recipe.state.add(float %sum, float %term)
%channel.next = add i32 %channel, 1
br label %channel.loop
store:
%product.index = add i64 %product.base.wide, %query.wide
%product.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i64 %product.index
store float %sum, ptr addrspace(3) %product.ptr, align 4
%query.next = add i32 %query, %block
br label %query.loop
exit:
ret void
}
define internal void @attention_tile_derivatives(ptr addrspace(1) nocapture readonly %context,
i32 %query.shared, i32 %key.shared, i32 %delta.shared, i32 %value.shared,
i32 %probability.shared, i32 %derivative.shared, i32 %product.shared,
i32 %query.base, i32 %key.base, i32 %query.count, i32 %key.count, i32 %tile.n,
i64 %head.job, i64 %length, i64 %statistics.denominator.base, i32 %head.width,
i8 %scale, i32 %lid, i32 %block, i64 %score.row, i32 %blocks, i32 %select.block, i1 %select) #1 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%query.shared.wide = zext i32 %query.shared to i64 %probability.shared.wide = zext i32 %probability.shared to i64 %derivative.shared.wide = zext i32 %derivative.shared to i64 %product.shared.wide = zext i32 %product.shared to i64 %query.base.wide = zext i32 %query.base to i64 %key.base.wide = zext i32 %key.base to i64 %tile.n.wide = zext i32 %tile.n to i64 %head.job.wide = add i64 %head.job, 0 %length.wide = add i64 %length, 0 %statistics.denominator.base.wide = add i64 %statistics.denominator.base, 0 %head.width.wide = zext i32 %head.width to i64
%pair.count = mul i32 %query.count, %key.count
%zero = call float @recipe.state.from.u1(i1 false)
br label %pair.loop
pair.loop:
%pair = phi i32 [ %lid, %entry ], [ %pair.next, %store ]
%pair.more = icmp ult i32 %pair, %pair.count
br i1 %pair.more, label %prepare, label %exit
prepare:
%query.local = udiv i32 %pair, %key.count
%key.local = urem i32 %pair, %key.count
%query.local.wide = zext i32 %query.local to i64 %key.local.wide = zext i32 %key.local to i64 %query = add i64 %query.base.wide, %query.local.wide
%key = add i64 %key.base.wide, %key.local.wide
%causal = icmp ule i64 %key, %query
br i1 %causal, label %selection, label %invalid
selection:
br i1 %select, label %selection.chosen, label %complete
selection.chosen:
%query.i32 = trunc i64 %query to i32 %key.i32 = trunc i64 %key to i32 %kept = call i1 @attention_selected(ptr addrspace(1) %context, i64 %score.row, i32 %blocks, i32 %select.block, i32 %query.i32, i32 %key.i32)
br i1 %kept, label %complete, label %invalid
complete:
%score.raw = call float @attention_tile_dot_state(i32 %query.local, i32 %key.local, i32 %head.width, i32 %query.shared, i32 %key.shared)
%scale.wide = call float @recipe.decode(i8 %scale)
%score = call float @recipe.state.div(float %score.raw, float %scale.wide)
%dp = call float @attention_tile_dot_state(i32 %query.local, i32 %key.local, i32 %head.width, i32 %delta.shared, i32 %value.shared)
%statistics.base = mul i64 %head.job.wide, %length.wide
%statistics.index = add i64 %statistics.base, %query
%maximum.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %statistics.index
%maximum = load i8, ptr addrspace(1) %maximum.ptr, align 1
%denominator.index = add i64 %statistics.denominator.base.wide, %statistics.index
%denominator.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %denominator.index
%denominator = load i8, ptr addrspace(1) %denominator.ptr, align 1
%maximum.wide = call float @recipe.decode(i8 %maximum)
%denominator.wide = call float @recipe.decode(i8 %denominator)
%centered = call float @recipe.state.sub(float %score, float %maximum.wide)
%exponential = call float @recipe.state.exp(float %centered)
%probability.wide = call float @recipe.state.div(float %exponential, float %denominator.wide)
%product.index = add i64 %product.shared.wide, %query.local.wide
%product.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i64 %product.index
%product = load float, ptr addrspace(3) %product.ptr, align 4
%product.wide = select i1 true, float %product, float %product
%dp.centered = call float @recipe.state.sub(float %dp, float %product.wide)
%derivative.wide = call float @recipe.state.mul(float %probability.wide, float %dp.centered)
br label %store
invalid:
br label %store
store:
%probability.value = phi float [ %probability.wide, %complete ], [ %zero, %invalid ]
%derivative.value = phi float [ %derivative.wide, %complete ], [ %zero, %invalid ]
%pair.row = mul i64 %query.local.wide, %tile.n.wide
%pair.local = add i64 %pair.row, %key.local.wide
%probability.index = add i64 %probability.shared.wide, %pair.local
%probability.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i64 %probability.index
store float %probability.value, ptr addrspace(3) %probability.ptr, align 4
%derivative.index = add i64 %derivative.shared.wide, %pair.local
%derivative.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i64 %derivative.index
store float %derivative.value, ptr addrspace(3) %derivative.ptr, align 4
%pair.next = add i32 %pair, %block
br label %pair.loop
exit:
ret void
}
; One query of llama.cpp's CPU attention, in its order: the query rounded to
; the cache type, each key's dot in four 16-lane accumulators of fused
; multiply-adds folded by halves, the keys walked in position order under an
; online softmax whose exponentials come from the CPU's libm, the values
; accumulated in the cache type (one fma, rounded, per key), the sum as
; fma(S, ms, vs), and the output the accumulator times one over the sum.
define internal void @attention_online_query(ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %kv.context,
i64 %row.stride, i64 %row.cache, i64 %output.row, i32 %head, i32 %position, i32 %heads, i32 %kv.heads, i32 %value.heads, i32 %width, i32 %length, i1 %unscaled, i1 %gate, i64 %gate.row, i32 %buffer.length, i32 %buffer.origin) #1 {
entry:
%q16 = alloca [256 x i8], align 1
%v16 = alloca [256 x i8], align 1
%lanes = alloca [64 x float], align 4
%state.zero = call float @recipe.state.from.u1(i1 false)
%one = call float @recipe.state.from.u1(i1 true)
%kv.group = udiv i32 %heads, %kv.heads
%value.group = udiv i32 %heads, %value.heads
%kv.head = udiv i32 %head, %kv.group
%value.head = udiv i32 %head, %value.group
%kv.channels = mul i32 %kv.heads, %width
%kv.plane = mul i32 %kv.channels, %length
%kv.plane.wide = zext i32 %kv.plane to i64
%length.wide = zext i32 %length to i64
%position.wide = zext i32 %position to i64
%width.state = call float @recipe.state.from.u32(i32 %width)
%root = call float @recipe.state.sqrt(float %width.state)
%q.scale.default = call float @recipe.state.div(float %one, float %root)
%q.scale = select i1 %unscaled, float %one, float %q.scale.default
%head.channel = mul i32 %head, %width
%kv.channel = mul i32 %kv.head, %width
%value.channel = mul i32 %value.head, %width
%head.channel.wide = zext i32 %head.channel to i64
%kv.channel.wide = zext i32 %kv.channel to i64
%value.channel.wide = zext i32 %value.channel to i64
%q.base = mul i64 %head.channel.wide, %length.wide
%q.row = add i64 %row.stride, %q.base
%k.base = mul i64 %kv.channel.wide, %length.wide
%k.row = add i64 %row.cache, %k.base
%v.base = mul i64 %value.channel.wide, %length.wide
%v.plane = add i64 %row.cache, %kv.plane.wide
%v.row = add i64 %v.plane, %v.base
%zero.kv = call i8 @recipe.kv.from.state(float %state.zero)
br label %q.loop
q.loop:
%q.d = phi i32 [ 0, %entry ], [ %q.d.next, %q.step ]
%q.more = icmp ult i32 %q.d, %width
br i1 %q.more, label %q.step, label %q.done
q.step:
%q.d.wide = zext i32 %q.d to i64
%q.channel.offset = mul i64 %q.d.wide, %length.wide
%q.index.base = add i64 %q.row, %q.channel.offset
%q.index = add i64 %q.index.base, %position.wide
%q.index.phys = call i64 @recipe.window.index(i64 %q.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%q.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %q.index.phys
%q.model = load i8, ptr addrspace(1) %q.ptr, align 1
%q.value = call float @recipe.state.from.model(i8 %q.model)
%q.scaled = call float @recipe.state.mul(float %q.value, float %q.scale)
%q.kv = call i8 @recipe.kv.from.state(float %q.scaled)
%q16.ptr = getelementptr i8, ptr %q16, i32 %q.d
store i8 %q.kv, ptr %q16.ptr, align 1
%v16.init.ptr = getelementptr i8, ptr %v16, i32 %q.d
store i8 %zero.kv, ptr %v16.init.ptr, align 1
%q.d.next = add i32 %q.d, 1
br label %q.loop
q.done:
%chunks = lshr i32 %width, 6
%np = shl i32 %chunks, 6
; Minus infinity as the state computes it: -1 / 0, so no literal names a type.
%minus.one = call float @recipe.state.neg(float %one)
%negative.infinity = call float @recipe.state.div(float %minus.one, float %state.zero)
br label %key.loop
key.loop:
%key = phi i32 [ 0, %q.done ], [ %key.next, %key.finish ]
%m = phi float [ %negative.infinity, %q.done ], [ %m.next, %key.finish ]
%s.sum = phi float [ %state.zero, %q.done ], [ %s.sum.next, %key.finish ]
%key.more = icmp ule i32 %key, %position
br i1 %key.more, label %lane.zero.loop, label %key.done
lane.zero.loop:
%lz = phi i32 [ 0, %key.loop ], [ %lz.next, %lane.zero.step ]
%lz.more = icmp ult i32 %lz, 64
br i1 %lz.more, label %lane.zero.step, label %chunk.loop
lane.zero.step:
%lz.ptr = getelementptr float, ptr %lanes, i32 %lz
store float %state.zero, ptr %lz.ptr, align 4
%lz.next = add i32 %lz, 1
br label %lane.zero.loop
chunk.loop:
%chunk = phi i32 [ 0, %lane.zero.loop ], [ %chunk.next, %chunk.advance ]
%chunk.more = icmp ult i32 %chunk, %np
br i1 %chunk.more, label %lane.loop, label %fold.loop
lane.loop:
%lane = phi i32 [ 0, %chunk.loop ], [ %lane.next, %lane.step ]
%lane.more = icmp ult i32 %lane, 64
br i1 %lane.more, label %lane.step, label %chunk.advance
lane.step:
%e = add i32 %chunk, %lane
%e.wide = zext i32 %e to i64
%e.offset = mul i64 %e.wide, %length.wide
%k.index.base = add i64 %k.row, %e.offset
%key.wide = zext i32 %key to i64
%k.index = add i64 %k.index.base, %key.wide
%k.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %k.index
%k.kv = load i8, ptr addrspace(1) %k.ptr, align 1
%k.value = call float @recipe.kv.to.state(i8 %k.kv)
%q16.lane.ptr = getelementptr i8, ptr %q16, i32 %e
%q.lane.kv = load i8, ptr %q16.lane.ptr, align 1
%q.lane.value = call float @recipe.kv.to.state(i8 %q.lane.kv)
%lane.ptr = getelementptr float, ptr %lanes, i32 %lane
%lane.value = load float, ptr %lane.ptr, align 4
%lane.value.next = call float @recipe.state.madd(float %lane.value, float %k.value, float %q.lane.value)
store float %lane.value.next, ptr %lane.ptr, align 4
%lane.next = add i32 %lane, 1
br label %lane.loop
chunk.advance:
%chunk.next = add i32 %chunk, 64
br label %chunk.loop
; The four accumulators fold to one: x0 += x2, x1 += x3, x0 += x1, then the
; sixteen lanes add by halves, 8, 4, 2, 1.
fold.loop:
%fold = phi i32 [ 0, %chunk.loop ], [ %fold.next, %fold.step ]
%fold.more = icmp ult i32 %fold, 16
br i1 %fold.more, label %fold.step, label %half.loop
fold.step:
%fold.x0.ptr = getelementptr float, ptr %lanes, i32 %fold
%fold.x1.index = add i32 %fold, 16
%fold.x1.ptr = getelementptr float, ptr %lanes, i32 %fold.x1.index
%fold.x2.index = add i32 %fold, 32
%fold.x2.ptr = getelementptr float, ptr %lanes, i32 %fold.x2.index
%fold.x3.index = add i32 %fold, 48
%fold.x3.ptr = getelementptr float, ptr %lanes, i32 %fold.x3.index
%fold.x0 = load float, ptr %fold.x0.ptr, align 4
%fold.x1 = load float, ptr %fold.x1.ptr, align 4
%fold.x2 = load float, ptr %fold.x2.ptr, align 4
%fold.x3 = load float, ptr %fold.x3.ptr, align 4
%fold.x02 = call float @recipe.state.add(float %fold.x0, float %fold.x2)
%fold.x13 = call float @recipe.state.add(float %fold.x1, float %fold.x3)
%fold.x = call float @recipe.state.add(float %fold.x02, float %fold.x13)
store float %fold.x, ptr %fold.x0.ptr, align 4
%fold.next = add i32 %fold, 1
br label %fold.loop
half.loop:
%half = phi i32 [ 8, %fold.loop ], [ %half.next, %half.done ]
%half.more = icmp ugt i32 %half, 0
br i1 %half.more, label %half.lane.loop, label %score.ready
half.lane.loop:
%half.lane = phi i32 [ 0, %half.loop ], [ %half.lane.next, %half.lane.step ]
%half.lane.more = icmp ult i32 %half.lane, %half
br i1 %half.lane.more, label %half.lane.step, label %half.done
half.lane.step:
%half.low.ptr = getelementptr float, ptr %lanes, i32 %half.lane
%half.high.index = add i32 %half.lane, %half
%half.high.ptr = getelementptr float, ptr %lanes, i32 %half.high.index
%half.low = load float, ptr %half.low.ptr, align 4
%half.high = load float, ptr %half.high.ptr, align 4
%half.sum = call float @recipe.state.add(float %half.low, float %half.high)
store float %half.sum, ptr %half.low.ptr, align 4
%half.lane.next = add i32 %half.lane, 1
br label %half.lane.loop
half.done:
%half.next = lshr i32 %half, 1
br label %half.loop
score.ready:
; The values past the last chunk of 64 add one product at a time.
%rest.ptr = getelementptr float, ptr %lanes, i32 0
%rest.start = load float, ptr %rest.ptr, align 4
br label %rest.loop
rest.loop:
%rest.e = phi i32 [ %np, %score.ready ], [ %rest.e.next, %rest.step ]
%rest.sum = phi float [ %rest.start, %score.ready ], [ %rest.sum.next, %rest.step ]
%rest.more = icmp ult i32 %rest.e, %width
br i1 %rest.more, label %rest.step, label %softmax
rest.step:
%rest.e.wide = zext i32 %rest.e to i64
%rest.offset = mul i64 %rest.e.wide, %length.wide
%rest.k.base = add i64 %k.row, %rest.offset
%rest.key.wide = zext i32 %key to i64
%rest.k.index = add i64 %rest.k.base, %rest.key.wide
%rest.k.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %rest.k.index
%rest.k.kv = load i8, ptr addrspace(1) %rest.k.ptr, align 1
%rest.k = call float @recipe.kv.to.state(i8 %rest.k.kv)
%rest.q.ptr = getelementptr i8, ptr %q16, i32 %rest.e
%rest.q.kv = load i8, ptr %rest.q.ptr, align 1
%rest.q = call float @recipe.kv.to.state(i8 %rest.q.kv)
%rest.product = call float @recipe.state.mul(float %rest.k, float %rest.q)
%rest.sum.next = call float @recipe.state.add(float %rest.sum, float %rest.product)
%rest.e.next = add i32 %rest.e, 1
br label %rest.loop
softmax:
%greater = call i1 @recipe.state.ogt(float %rest.sum, float %m)
%m.next = select i1 %greater, float %rest.sum, float %m
%m.drop = call float @recipe.state.sub(float %m, float %m.next)
%ms.raw = call float @recipe.libm.exp(float %m.drop)
%ms = select i1 %greater, float %ms.raw, float %one
%s.drop = call float @recipe.state.sub(float %rest.sum, float %m.next)
%vs.raw = call float @recipe.libm.exp(float %s.drop)
%vs = select i1 %greater, float %one, float %vs.raw
br label %value.loop
value.loop:
%vd = phi i32 [ 0, %softmax ], [ %vd.next, %value.step ]
%vd.more = icmp ult i32 %vd, %width
br i1 %vd.more, label %value.step, label %key.finish
value.step:
%v16.ptr = getelementptr i8, ptr %v16, i32 %vd
%acc.kv = load i8, ptr %v16.ptr, align 1
%acc = call float @recipe.kv.to.state(i8 %acc.kv)
%acc.scaled.raw = call float @recipe.state.mul(float %acc, float %ms)
%acc.scaled.kv = call i8 @recipe.kv.from.state(float %acc.scaled.raw)
%acc.scaled = call float @recipe.kv.to.state(i8 %acc.scaled.kv)
%acc.base = select i1 %greater, float %acc.scaled, float %acc
%vd.wide = zext i32 %vd to i64
%vd.offset = mul i64 %vd.wide, %length.wide
%v.index.base = add i64 %v.row, %vd.offset
%v.key.wide = zext i32 %key to i64
%v.index = add i64 %v.index.base, %v.key.wide
%v.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %v.index
%v.kv = load i8, ptr addrspace(1) %v.ptr, align 1
%v.value = call float @recipe.kv.to.state(i8 %v.kv)
%acc.next.raw = call float @recipe.state.madd(float %acc.base, float %v.value, float %vs)
%acc.next.kv = call i8 @recipe.kv.from.state(float %acc.next.raw)
store i8 %acc.next.kv, ptr %v16.ptr, align 1
%vd.next = add i32 %vd, 1
br label %value.loop
key.finish:
%s.sum.next = call float @recipe.state.madd(float %vs, float %s.sum, float %ms)
%key.next = add i32 %key, 1
br label %key.loop
key.done:
%inverse = call float @recipe.state.div(float %one, float %s.sum)
br label %out.loop
out.loop:
%od = phi i32 [ 0, %key.done ], [ %od.next, %out.store ]
%od.more = icmp ult i32 %od, %width
br i1 %od.more, label %out.step, label %done
out.step:
%out.acc.ptr = getelementptr i8, ptr %v16, i32 %od
%out.acc.kv = load i8, ptr %out.acc.ptr, align 1
%out.acc = call float @recipe.kv.to.state(i8 %out.acc.kv)
%out.value = call float @recipe.state.mul(float %out.acc, float %inverse)
%od.wide = zext i32 %od to i64
%out.offset = mul i64 %od.wide, %length.wide
%out.local.base = add i64 %q.base, %out.offset
%out.local = add i64 %out.local.base, %position.wide
%out.index = add i64 %output.row, %out.local
%out.model = call i8 @recipe.model.from.state(float %out.value)
br i1 %gate, label %out.gate, label %out.store
out.gate:
%gate.index = add i64 %gate.row, %out.local
%gate.index.phys = call i64 @recipe.window.index(i64 %gate.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %gate.index.phys
%gate.value = load i8, ptr addrspace(1) %gate.ptr, align 1
%gate.factor = call i8 @recipe.sigmoid(i8 %gate.value)
%out.gated = call i8 @recipe.mul(i8 %out.model, i8 %gate.factor)
br label %out.store
out.store:
%out.result = phi i8 [ %out.model, %out.step ], [ %out.gated, %out.gate ]
%out.index.phys = call i64 @recipe.window.index(i64 %out.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%out.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %out.index.phys
store i8 %out.result, ptr addrspace(1) %out.ptr, align 1
%od.next = add i32 %od, 1
br label %out.loop
done:
ret void
}
define internal void @attention_forward_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %weights,
ptr addrspace(1) nocapture writeonly %output, ptr addrspace(1) %context, ptr addrspace(1) %kv.context, i1 %carry,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %query.begin, i32 %query.span, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, i8 %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, float %index.base, i1 %online, i32 %buffer.length, i32 %buffer.origin, i32 %query.capacity ) #3 { entry:
%query.capacity.wide = zext i32 %query.capacity to i64
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%head.width.double = call i8 @recipe.from.u32(i32 %head.width)
%scale.default = call i8 @recipe.sqrt(i8 %head.width.double)
%attention.zero = call i8 @recipe.from.u32(i32 0)
%attention.one = call i8 @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(i8 %attention.zero, i8 %epsilon)
%scale = select i1 %unscaled, i8 %attention.one, i8 %scale.default
%kv.group = udiv i32 %heads, %kv.heads
%value.group = udiv i32 %heads, %value.heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%value.channels = mul i32 %value.heads, %head.width
%value.plane = mul i32 %value.channels, %length
%kv.planes = add i32 %kv.plane, %value.plane
%value.plane.base = add i32 %from, %kv.plane
%index.query.channels = mul i32 %index.heads, %index.width
%index.channels = add i32 %index.query.channels, %index.width
%index.plane = mul i32 %index.channels, %length
%gate.plane = select i1 %gate, i32 %from, i32 0
%index.query.base = add i32 %from, %kv.planes
%gate.base = add i32 %index.query.base, 0
%row.stride = add i32 %gate.base, %gate.plane
%select = icmp ne i32 %select.block, 0
%block.divisor = select i1 %select, i32 %select.block, i32 1
%blocks.numerator = add i32 %length, %block.divisor
%blocks.less = sub i32 %blocks.numerator, 1
%blocks.full = udiv i32 %blocks.less, %block.divisor
%blocks = select i1 %select, i32 %blocks.full, i32 0
%score.stride = mul i32 %blocks, 2
%tile.m.less.one = sub i32 %tile.m, 1
%query.tiles.rounded = add i32 %query.span, %tile.m.less.one
%query.tiles = udiv i32 %query.tiles.rounded, %tile.m
%head.jobs = mul i32 %rows, %heads
%statistics.plane = mul i32 %head.jobs, %query.capacity
%representative.base = mul i32 %statistics.plane, 2
%representative.stride = mul i32 %blocks, %index.width
%representative.total = mul i32 %representative.stride, %rows
%score.base = add i32 %representative.base, %representative.total
%rows.global = zext i32 %rows to i64 %from.global = zext i32 %from to i64 %channels.global = zext i32 %channels to i64 %heads.global = zext i32 %heads to i64 %length.global = zext i32 %length to i64 %head.width.global = zext i32 %head.width to i64 %blocks.global = zext i32 %blocks to i64 %index.width.global = zext i32 %index.width to i64 %block.divisor.global = zext i32 %block.divisor to i64 %kv.heads.global = zext i32 %kv.heads to i64 %index.heads.global = zext i32 %index.heads to i64
%kv.channels.global = mul i64 %kv.heads.global, %head.width.global %kv.plane.global = mul i64 %kv.channels.global, %length.global %value.heads.global = zext i32 %value.heads to i64 %value.channels.global = mul i64 %value.heads.global, %head.width.global %value.plane.global = mul i64 %value.channels.global, %length.global %kv.planes.global = add i64 %kv.plane.global, %value.plane.global %value.plane.base.global = add i64 %from.global, %kv.plane.global
%index.query.channels.global = mul i64 %index.heads.global, %index.width.global %index.channels.global = add i64 %index.query.channels.global, %index.width.global %index.plane.global = mul i64 %index.channels.global, %length.global
%gate.plane.global = select i1 %gate, i64 %from.global, i64 0 %index.query.base.global = add i64 %from.global, %kv.planes.global %gate.base.global = add i64 %index.query.base.global, 0 %row.stride.global = add i64 %gate.base.global, %gate.plane.global
%blocks.numerator.global = add i64 %length.global, %block.divisor.global %blocks.less.global = sub i64 %blocks.numerator.global, 1 %blocks.full.global = udiv i64 %blocks.less.global, %block.divisor.global %blocks.selected.global = select i1 %select, i64 %blocks.full.global, i64 0
%score.stride.global = mul i64 %blocks.selected.global, 2 %head.jobs.global = mul i64 %rows.global, %heads.global %statistics.plane.global = mul i64 %head.jobs.global, %query.capacity.wide %representative.base.global = mul i64 %statistics.plane.global, 2 %representative.stride.global = mul i64 %blocks.selected.global, %index.width.global %representative.total.global = mul i64 %representative.stride.global, %rows.global %score.base.global = add i64 %representative.base.global, %representative.total.global
%score.base.wide = add i64 %score.base.global, 0
%length.wide = add i64 %length.global, 0
%score.stride.wide = add i64 %score.stride.global, 0
%score.row.stride = mul i64 %query.capacity.wide, %score.stride.wide
%jobs = mul i32 %head.jobs, %query.tiles
%query.values = mul i32 %tile.m, %head.width
%key.values = mul i32 %tile.n, %head.width
%key.base.shared = add i32 0, %query.values
%value.base.shared = add i32 %key.base.shared, %key.values
%score.base.shared = add i32 %value.base.shared, %key.values
%score.values = mul i32 %tile.m, %tile.n
%probability.base.shared = add i32 %score.base.shared, %score.values
%accumulator.base.shared = add i32 %probability.base.shared, %score.values
%maximum.base.shared = add i32 %accumulator.base.shared, %query.values
%denominator.base.shared = add i32 %maximum.base.shared, %tile.m
%online.global = mul i32 %group, %block
%online.id = add i32 %online.global, %lid
%online.cache.channels = add i32 %kv.channels, %value.channels
%online.cache.per.row = mul i32 %online.cache.channels, %query.span
%online.cache.total = mul i32 %online.cache.per.row, %rows
br i1 %online, label %online.cache.loop, label %job.loop
; The online order: the window's keys and values enter the cache in the cache
; type first, then one thread per row, head and query walks its keys.
online.cache.loop:
%oc = phi i32 [ %online.id, %entry ], [ %oc.next, %online.cache.step ]
%oc.more = icmp ult i32 %oc, %online.cache.total
br i1 %oc.more, label %online.cache.step, label %online.cache.done
online.cache.step:
%oc.local = urem i32 %oc, %query.span
%oc.rest = udiv i32 %oc, %query.span
%oc.channel = urem i32 %oc.rest, %online.cache.channels
%oc.row = udiv i32 %oc.rest, %online.cache.channels
%oc.position = add i32 %query.begin, %oc.local
%oc.is.value = icmp uge i32 %oc.channel, %kv.channels
%oc.value.channel = sub i32 %oc.channel, %kv.channels
%oc.plane.channel = select i1 %oc.is.value, i32 %oc.value.channel, i32 %oc.channel
%oc.source.plane = select i1 %oc.is.value, i32 %value.plane.base, i32 %from
%oc.cache.plane = select i1 %oc.is.value, i32 %kv.plane, i32 0
%oc.row.wide = zext i32 %oc.row to i64
%oc.source.row = mul i64 %oc.row.wide, %row.stride.global
%oc.cache.row = mul i64 %oc.row.wide, %kv.planes.global
%oc.plane.channel.wide = zext i32 %oc.plane.channel to i64
%oc.channel.base = mul i64 %oc.plane.channel.wide, %length.global
%oc.position.wide = zext i32 %oc.position to i64
%oc.local.index = add i64 %oc.channel.base, %oc.position.wide
%oc.source.plane.wide = zext i32 %oc.source.plane to i64
%oc.source.base = add i64 %oc.source.row, %oc.source.plane.wide
%oc.source.index = add i64 %oc.source.base, %oc.local.index
%oc.source.index.phys = call i64 @recipe.window.index(i64 %oc.source.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%oc.source.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %oc.source.index.phys
%oc.value = load i8, ptr addrspace(1) %oc.source.ptr, align 1
%oc.cache.plane.wide = zext i32 %oc.cache.plane to i64
%oc.cache.base = add i64 %oc.cache.row, %oc.cache.plane.wide
%oc.cache.index = add i64 %oc.cache.base, %oc.local.index
%oc.cache.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %oc.cache.index
%oc.kv = call i8 @recipe.kv.encode(i8 %oc.value)
store i8 %oc.kv, ptr addrspace(1) %oc.cache.ptr, align 1
%oc.next = add i32 %oc, %threads
br label %online.cache.loop
online.cache.done:
call void @grid_barrier(i32 %threads)
%online.queries.per.row = mul i32 %heads, %query.span
%online.queries = mul i32 %online.queries.per.row, %rows
br label %online.query.loop
online.query.loop:
%oq = phi i32 [ %online.id, %online.cache.done ], [ %oq.next, %online.query.step ]
%oq.more = icmp ult i32 %oq, %online.queries
br i1 %oq.more, label %online.query.step, label %exit
online.query.step:
%oq.local = urem i32 %oq, %query.span
%oq.rest = udiv i32 %oq, %query.span
%oq.head = urem i32 %oq.rest, %heads
%oq.row = udiv i32 %oq.rest, %heads
%oq.position = add i32 %query.begin, %oq.local
%oq.row.wide = zext i32 %oq.row to i64
%oq.row.stride = mul i64 %oq.row.wide, %row.stride.global
%oq.row.cache = mul i64 %oq.row.wide, %kv.planes.global
%oq.output.row = mul i64 %oq.row.wide, %from.global
%oq.gate.row = add i64 %oq.row.stride, %gate.base.global
call void @attention_online_query(ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %kv.context, i64 %oq.row.stride, i64 %oq.row.cache, i64 %oq.output.row, i32 %oq.head, i32 %oq.position, i32 %heads, i32 %kv.heads, i32 %value.heads, i32 %head.width, i32 %length, i1 %unscaled, i1 %gate, i64 %oq.gate.row, i32 %buffer.length, i32 %buffer.origin)
%oq.next = add i32 %oq, %threads
br label %online.query.loop
job.loop:
%job = phi i32 [ %group, %entry ], [ %job.next, %job.finish ]
%job.more = icmp ult i32 %job, %jobs
br i1 %job.more, label %job.prepare, label %exit
job.prepare:
%query.tile = urem i32 %job, %query.tiles
%head.job = udiv i32 %job, %query.tiles
%head = urem i32 %head.job, %heads
%row = udiv i32 %head.job, %heads
%query.local.base = mul i32 %query.tile, %tile.m
%query.base = add i32 %query.local.base, %query.begin
%query.remaining = sub i32 %query.span, %query.local.base
%query.full = icmp ult i32 %query.remaining, %tile.m
%query.count = select i1 %query.full, i32 %query.remaining, i32 %tile.m
%query.last = add i32 %query.base, %query.count
%row.base = mul i32 %row, %row.stride
%head.start = mul i32 %head, %head.width
%kv.head = udiv i32 %head, %kv.group
%kv.head.start = mul i32 %kv.head, %head.width
%value.head = udiv i32 %head, %value.group
%value.head.start = mul i32 %value.head, %head.width
%row.wide = zext i32 %row to i64
%row.base.wide = mul i64 %row.wide, %row.stride.global
%head.start.wide = zext i32 %head.start to i64
%score.row = mul i64 %row.wide, %score.row.stride
%score.row.base = add i64 %score.base.wide, %score.row
%active.query.values = mul i32 %query.count, %head.width
br label %query.stage.loop
query.stage.loop:
%query.p = phi i32 [ %lid, %job.prepare ], [ %query.p.next, %query.stage.step ]
%query.p.more = icmp ult i32 %query.p, %active.query.values
br i1 %query.p.more, label %query.stage.step, label %statistics.init.loop
query.stage.step:
%query.local = udiv i32 %query.p, %head.width
%query.channel.local = urem i32 %query.p, %head.width
%query.position = add i32 %query.base, %query.local
%query.channel = add i32 %head.start, %query.channel.local
%query.channel.wide = zext i32 %query.channel to i64 %query.channel.base = mul i64 %query.channel.wide, %length.global
%query.local.wide = zext i32 %query.local to i64 %query.position.wide = zext i32 %query.position to i64 %query.input.local = add i64 %query.channel.base, %query.position.wide
%query.input.index = add i64 %row.base.wide, %query.input.local
%query.input.index.phys = call i64 @recipe.window.index(i64 %query.input.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%query.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %query.input.index.phys
%query.value = load i8, ptr addrspace(1) %query.input.ptr, align 1
%query.shared.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %query.p
store i8 %query.value, ptr addrspace(3) %query.shared.ptr, align 1
%accumulator.index = add i32 %accumulator.base.shared, %query.p
%accumulator.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %accumulator.index
store i8 0, ptr addrspace(3) %accumulator.ptr, align 1
%query.p.next = add i32 %query.p, %block
br label %query.stage.loop
statistics.init.loop:
%statistics.q = phi i32 [ %lid, %query.stage.loop ], [ %statistics.q.next, %statistics.init.step ]
%statistics.q.more = icmp ult i32 %statistics.q, %query.count
br i1 %statistics.q.more, label %statistics.init.step, label %query.stage.done
statistics.init.step:
%maximum.index.init = add i32 %maximum.base.shared, %statistics.q
%maximum.ptr.init = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %maximum.index.init
store i8 254, ptr addrspace(3) %maximum.ptr.init, align 1
%denominator.index.init = add i32 %denominator.base.shared, %statistics.q
%denominator.ptr.init = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %denominator.index.init
store i8 0, ptr addrspace(3) %denominator.ptr.init, align 1
%statistics.q.next = add i32 %statistics.q, %block
br label %statistics.init.loop
query.stage.done:
call void @llvm.nvvm.barrier0()
br label %query.norm.done
query.norm.done:
br label %key.tile.loop
key.tile.loop:
%key.tile.base = phi i32 [ 0, %query.norm.done ], [ %key.tile.next, %key.tile.advance ]
%key.tile.more = icmp ult i32 %key.tile.base, %query.last
br i1 %key.tile.more, label %key.tile.prepare, label %output.loop
key.tile.prepare:
%key.remaining = sub i32 %query.last, %key.tile.base
%key.full = icmp ult i32 %key.remaining, %tile.n
%key.count = select i1 %key.full, i32 %key.remaining, i32 %tile.n
%active.key.values = mul i32 %key.count, %head.width
br i1 %select, label %tile.scan.prepare, label %key.stage.loop
tile.scan.prepare:
%tile.first.block = udiv i32 %key.tile.base, %select.block
%tile.stop = add i32 %key.tile.base, %key.count
%tile.stop.less = sub i32 %tile.stop, 1
%tile.last.block = udiv i32 %tile.stop.less, %select.block
br label %tile.scan.loop
tile.scan.loop:
%tile.scan.q = phi i32 [ 0, %tile.scan.prepare ], [ %tile.scan.q.next, %tile.scan.block.done ]
%tile.scan.more = icmp ult i32 %tile.scan.q, %query.count
br i1 %tile.scan.more, label %tile.scan.query, label %key.tile.advance
tile.scan.query:
%tile.scan.query.index = add i32 %query.base, %tile.scan.q
br label %tile.scan.block.loop
tile.scan.block.loop:
%tile.scan.b = phi i32 [ %tile.first.block, %tile.scan.query ], [ %tile.scan.b.next, %tile.scan.block.advance ]
%tile.scan.block.more = icmp ule i32 %tile.scan.b, %tile.last.block
br i1 %tile.scan.block.more, label %tile.scan.block.step, label %tile.scan.block.done
tile.scan.block.step:
%tile.scan.block.start = mul i32 %tile.scan.b, %select.block
%tile.scan.before = icmp ult i32 %tile.scan.block.start, %key.tile.base
%tile.scan.key = select i1 %tile.scan.before, i32 %key.tile.base, i32 %tile.scan.block.start
%tile.scan.causal = icmp ule i32 %tile.scan.key, %tile.scan.query.index
%tile.scan.kept = call i1 @attention_selected(ptr addrspace(1) %context, i64 %score.row.base, i32 %blocks, i32 %select.block, i32 %tile.scan.query.index, i32 %tile.scan.key)
%tile.scan.hit = and i1 %tile.scan.causal, %tile.scan.kept
br i1 %tile.scan.hit, label %key.stage.loop, label %tile.scan.block.advance
tile.scan.block.advance:
%tile.scan.b.next = add i32 %tile.scan.b, 1
br label %tile.scan.block.loop
tile.scan.block.done:
%tile.scan.q.next = add i32 %tile.scan.q, 1
br label %tile.scan.loop
key.stage.loop:
%key.p = phi i32 [ %lid, %key.tile.prepare ], [ %lid, %tile.scan.block.step ], [ %key.p.next, %key.loaded ]
%key.p.more = icmp ult i32 %key.p, %active.key.values
br i1 %key.p.more, label %key.stage.step, label %key.stage.done
key.stage.step:
%key.local = udiv i32 %key.p, %head.width
%key.channel.local = urem i32 %key.p, %head.width
%key.position = add i32 %key.tile.base, %key.local
%key.channel = add i32 %kv.head.start, %key.channel.local
%key.channel.wide = zext i32 %key.channel to i64 %key.channel.base = mul i64 %key.channel.wide, %length.global
%key.local.wide = zext i32 %key.local to i64 %key.position.wide = zext i32 %key.position to i64 %key.input.local = add i64 %key.channel.base, %key.position.wide
%value.channel = add i32 %value.head.start, %key.channel.local
%value.channel.wide = zext i32 %value.channel to i64 %value.channel.base = mul i64 %value.channel.wide, %length.global %value.input.local = add i64 %value.channel.base, %key.position.wide
%key.plane = add i64 %row.base.wide, %from.global
%key.input.index = add i64 %key.plane, %key.input.local
%key.past = icmp ult i32 %key.position, %query.begin
%key.use.cache = and i1 %carry, %key.past
br i1 %key.use.cache, label %key.cache.load, label %key.source.load
key.cache.load:
%key.cache.row = mul i64 %row.wide, %kv.planes.global
%key.cache.index = add i64 %key.cache.row, %key.input.local
%key.cache.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %key.cache.index
%key.cache.kv = load i8, ptr addrspace(1) %key.cache.ptr, align 1
%key.cache.value = call i8 @recipe.kv.decode(i8 %key.cache.kv)
%value.cache.index = add i64 %key.cache.row, %kv.plane.global
%value.cache.index.final = add i64 %value.cache.index, %value.input.local
%value.cache.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %value.cache.index.final
%value.cache.kv = load i8, ptr addrspace(1) %value.cache.ptr, align 1
%value.cache.value = call i8 @recipe.kv.decode(i8 %value.cache.kv)
br label %key.loaded
key.source.load:
%key.input.index.phys = call i64 @recipe.window.index(i64 %key.input.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%key.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %key.input.index.phys
%key.source.value = load i8, ptr addrspace(1) %key.input.ptr, align 1
%value.row = add i64 %row.base.wide, %value.plane.base.global
%value.input.index = add i64 %value.row, %value.input.local
%value.input.index.phys = call i64 @recipe.window.index(i64 %value.input.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%value.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %value.input.index.phys
%value.source.value = load i8, ptr addrspace(1) %value.input.ptr, align 1
br i1 %carry, label %key.cache.store, label %key.loaded.source
key.cache.store:
%key.store.row = mul i64 %row.wide, %kv.planes.global
%key.store.index = add i64 %key.store.row, %key.input.local
%key.store.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %key.store.index
%key.store.kv = call i8 @recipe.kv.encode(i8 %key.source.value)
store i8 %key.store.kv, ptr addrspace(1) %key.store.ptr, align 1
%value.store.index = add i64 %key.store.row, %kv.plane.global
%value.store.index.final = add i64 %value.store.index, %value.input.local
%value.store.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i64 %value.store.index.final
%value.store.kv = call i8 @recipe.kv.encode(i8 %value.source.value)
store i8 %value.store.kv, ptr addrspace(1) %value.store.ptr, align 1
; A carried key is read back the way the cache holds it, so the window's
; own keys meet the same rounding as the past ones and a prefill gives
; the same bits however it is windowed.
%key.store.value = call i8 @recipe.kv.decode(i8 %key.store.kv)
%value.store.value = call i8 @recipe.kv.decode(i8 %value.store.kv)
br label %key.loaded
key.loaded.source:
br label %key.loaded
key.loaded:
%key.value = phi i8 [ %key.cache.value, %key.cache.load ], [ %key.source.value, %key.loaded.source ], [ %key.store.value, %key.cache.store ]
%value.value = phi i8 [ %value.cache.value, %key.cache.load ], [ %value.source.value, %key.loaded.source ], [ %value.store.value, %key.cache.store ]
%key.shared.index = add i32 %key.base.shared, %key.p
%key.shared.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %key.shared.index
store i8 %key.value, ptr addrspace(3) %key.shared.ptr, align 1
%value.shared.index = add i32 %value.base.shared, %key.p
%value.shared.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %value.shared.index
store i8 %value.value, ptr addrspace(3) %value.shared.ptr, align 1
%key.p.next = add i32 %key.p, %block
br label %key.stage.loop
key.stage.done:
call void @llvm.nvvm.barrier0()
br label %key.norm.done
key.norm.done:
%score.count = mul i32 %query.count, %key.count
br label %score.loop
score.loop:
%score.p = phi i32 [ %lid, %key.norm.done ], [ %score.p.next, %score.store ]
%score.p.more = icmp ult i32 %score.p, %score.count
br i1 %score.p.more, label %score.prepare, label %score.done
score.prepare:
%score.query.local = udiv i32 %score.p, %key.count
%score.key.local = urem i32 %score.p, %key.count
%score.query = add i32 %query.base, %score.query.local
%score.key = add i32 %key.tile.base, %score.key.local
%score.causal = icmp ule i32 %score.key, %score.query
br i1 %score.causal, label %score.selection, label %score.invalid
score.selection:
br i1 %select, label %score.selection.chosen, label %score.complete
score.selection.chosen:
%score.kept = call i1 @attention_selected(ptr addrspace(1) %context, i64 %score.row.base, i32 %blocks, i32 %select.block, i32 %score.query, i32 %score.key)
br i1 %score.kept, label %score.complete, label %score.invalid
score.complete:
%score.scaled = call i8 @attention_tile_score(i32 %score.query.local, i32 %score.key.local, i32 %head.width, i32 %key.base.shared, i8 %scale)
br label %score.store
score.invalid:
br label %score.store
score.store:
%score.value = phi i8 [ %score.scaled, %score.complete ], [ 254, %score.invalid ]
%score.shared.row = mul i32 %score.query.local, %tile.n
%score.shared.local = add i32 %score.shared.row, %score.key.local
%score.shared.index = add i32 %score.base.shared, %score.shared.local
%score.shared.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %score.shared.index
store i8 %score.value, ptr addrspace(3) %score.shared.ptr, align 1
%score.p.next = add i32 %score.p, %block
br label %score.loop
score.done:
call void @llvm.nvvm.barrier0()
br label %softmax.loop
softmax.loop:
%softmax.query = phi i32 [ %lid, %score.done ], [ %softmax.query.next, %softmax.store ]
%softmax.more = icmp ult i32 %softmax.query, %query.count
br i1 %softmax.more, label %maximum.load, label %softmax.done
maximum.load:
%maximum.index = add i32 %maximum.base.shared, %softmax.query
%maximum.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %maximum.index
%maximum.old = load i8, ptr addrspace(3) %maximum.ptr, align 1
%denominator.index = add i32 %denominator.base.shared, %softmax.query
%denominator.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %denominator.index
%denominator.old = load i8, ptr addrspace(3) %denominator.ptr, align 1
; Fold keys in sequence order, including each rescale and storage rounding.
; Physical tile boundaries only determine staging, not arithmetic grouping.
; Once consumed, a score slot carries that key's rescale for the value fold.
br label %probability.loop
probability.loop:
%probability.key = phi i32 [ 0, %maximum.load ], [ %probability.key.next, %probability.step ]
%maximum.value = phi i8 [ %maximum.old, %maximum.load ], [ %maximum.next, %probability.step ]
%denominator.value = phi i8 [ %denominator.old, %maximum.load ], [ %denominator.next.model, %probability.step ]
%probability.more = icmp ult i32 %probability.key, %key.count
br i1 %probability.more, label %probability.step, label %softmax.store
probability.step:
%probability.row = mul i32 %softmax.query, %tile.n
%probability.local = add i32 %probability.row, %probability.key
%probability.score.index = add i32 %score.base.shared, %probability.local
%probability.score.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %probability.score.index
%probability.score = load i8, ptr addrspace(3) %probability.score.ptr, align 1
%maximum.larger = call i1 @recipe.ogt(i8 %probability.score, i8 %maximum.value)
%maximum.next = select i1 %maximum.larger, i8 %probability.score, i8 %maximum.value
%maximum.scored = call i1 @recipe.ogt(i8 %maximum.next, i8 254)
%maximum.safe = select i1 %maximum.scored, i8 %maximum.next, i8 0
%maximum.old.centered = call i8 @recipe.sub(i8 %maximum.value, i8 %maximum.safe)
%old.rescale = call i8 @recipe.exp(i8 %maximum.old.centered)
store i8 %old.rescale, ptr addrspace(3) %probability.score.ptr, align 1
%probability.centered = call i8 @recipe.sub(i8 %probability.score, i8 %maximum.safe)
%probability.value = call i8 @recipe.exp(i8 %probability.centered)
%probability.index = add i32 %probability.base.shared, %probability.local
%probability.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %probability.index
store i8 %probability.value, ptr addrspace(3) %probability.ptr, align 1
%denominator.old.wide = call float @recipe.decode(i8 %denominator.value)
%old.rescale.wide = call float @recipe.decode(i8 %old.rescale)
%denominator.rescaled = call float @recipe.state.mul(float %denominator.old.wide, float %old.rescale.wide)
%probability.wide = call float @recipe.decode(i8 %probability.value)
%denominator.next = call float @recipe.state.add(float %denominator.rescaled, float %probability.wide)
%denominator.next.model = call i8 @recipe.encode(float %denominator.next)
%probability.key.next = add i32 %probability.key, 1
br label %probability.loop
softmax.store:
store i8 %maximum.value, ptr addrspace(3) %maximum.ptr, align 1
store i8 %denominator.value, ptr addrspace(3) %denominator.ptr, align 1
%softmax.query.next = add i32 %softmax.query, %block
br label %softmax.loop
softmax.done:
call void @llvm.nvvm.barrier0()
br label %accumulate.loop
accumulate.loop:
%accumulate.p = phi i32 [ %lid, %softmax.done ], [ %accumulate.p.next, %accumulate.store ]
%accumulate.more = icmp ult i32 %accumulate.p, %active.query.values
br i1 %accumulate.more, label %accumulate.prepare, label %accumulate.done
accumulate.prepare:
%accumulate.query = udiv i32 %accumulate.p, %head.width
%accumulate.channel = urem i32 %accumulate.p, %head.width
%accumulate.index = add i32 %accumulator.base.shared, %accumulate.p
%accumulate.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %accumulate.index
%accumulate.old = load i8, ptr addrspace(3) %accumulate.ptr, align 1
br label %accumulate.key.loop
accumulate.key.loop:
%accumulate.key = phi i32 [ 0, %accumulate.prepare ], [ %accumulate.key.next, %accumulate.key.step ]
%accumulate.value = phi i8 [ %accumulate.old, %accumulate.prepare ], [ %accumulate.next, %accumulate.key.step ]
%accumulate.key.more = icmp ult i32 %accumulate.key, %key.count
br i1 %accumulate.key.more, label %accumulate.key.step, label %accumulate.store
accumulate.key.step:
%accumulate.probability.row = mul i32 %accumulate.query, %tile.n
%accumulate.probability.local = add i32 %accumulate.probability.row, %accumulate.key
%accumulate.probability.index = add i32 %probability.base.shared, %accumulate.probability.local
%accumulate.probability.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %accumulate.probability.index
%accumulate.probability = load i8, ptr addrspace(3) %accumulate.probability.ptr, align 1
%accumulate.value.row = mul i32 %accumulate.key, %head.width
%accumulate.value.local = add i32 %accumulate.value.row, %accumulate.channel
%accumulate.value.index = add i32 %value.base.shared, %accumulate.value.local
%accumulate.value.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %accumulate.value.index
%accumulate.v = load i8, ptr addrspace(3) %accumulate.value.ptr, align 1
%accumulate.weighted = call i8 @recipe.mul(i8 %accumulate.probability, i8 %accumulate.v)
%accumulate.rescale.index = add i32 %score.base.shared, %accumulate.probability.local
%accumulate.rescale.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %accumulate.rescale.index
%accumulate.rescale = load i8, ptr addrspace(3) %accumulate.rescale.ptr, align 1
%accumulate.rescaled = call i8 @recipe.mul(i8 %accumulate.value, i8 %accumulate.rescale)
%accumulate.next = call i8 @recipe.add(i8 %accumulate.rescaled, i8 %accumulate.weighted)
%accumulate.key.next = add i32 %accumulate.key, 1
br label %accumulate.key.loop
accumulate.store:
store i8 %accumulate.value, ptr addrspace(3) %accumulate.ptr, align 1
%accumulate.p.next = add i32 %accumulate.p, %block
br label %accumulate.loop
accumulate.done:
call void @llvm.nvvm.barrier0()
br label %key.tile.advance
key.tile.advance:
%key.tile.next = add i32 %key.tile.base, %tile.n
br label %key.tile.loop
output.loop:
%output.p = phi i32 [ %lid, %key.tile.loop ], [ %output.p.next, %output.plain ]
%output.more = icmp ult i32 %output.p, %active.query.values
br i1 %output.more, label %output.store, label %output.done
output.store:
%output.query.local = udiv i32 %output.p, %head.width
%output.channel.local = urem i32 %output.p, %head.width
%output.accumulator.index = add i32 %accumulator.base.shared, %output.p
%output.accumulator.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %output.accumulator.index
%output.accumulator = load i8, ptr addrspace(3) %output.accumulator.ptr, align 1
%output.denominator.index = add i32 %denominator.base.shared, %output.query.local
%output.denominator.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %output.denominator.index
%output.denominator = load i8, ptr addrspace(3) %output.denominator.ptr, align 1
%attention = call i8 @recipe.div(i8 %output.accumulator, i8 %output.denominator)
%output.query = add i32 %query.base, %output.query.local
%output.statistics.owner = icmp eq i32 %output.channel.local, 0
br i1 %output.statistics.owner, label %output.statistics.store, label %output.value.store
output.statistics.store:
%output.maximum.index = add i32 %maximum.base.shared, %output.query.local
%output.maximum.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %output.maximum.index
%output.maximum = load i8, ptr addrspace(3) %output.maximum.ptr, align 1
%output.statistics.head.job.wide = zext i32 %head.job to i64 %output.statistics.query.wide = zext i32 %output.query to i64 %output.statistics.head.base = mul i64 %output.statistics.head.job.wide, %length.global
%output.statistics.index = add i64 %output.statistics.head.base, %output.statistics.query.wide
%output.statistics.maximum.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %output.statistics.index
store i8 %output.maximum, ptr addrspace(1) %output.statistics.maximum.ptr, align 1
%output.statistics.denominator.index = add i64 %statistics.plane.global, %output.statistics.index
%output.statistics.denominator.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %output.statistics.denominator.index
store i8 %output.denominator, ptr addrspace(1) %output.statistics.denominator.ptr, align 1
br label %output.value.store
output.value.store:
%output.channel = add i32 %head.start, %output.channel.local
%output.channel.wide = zext i32 %output.channel to i64 %output.channel.base = mul i64 %output.channel.wide, %length.global
%output.query.wide = zext i32 %output.query to i64 %output.local = add i64 %output.channel.base, %output.query.wide
%output.row.base = mul i64 %row.wide, %from.global
%output.index = add i64 %output.row.base, %output.local
%output.index.phys = call i64 @recipe.window.index(i64 %output.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.index.phys
br i1 %gate, label %output.gate, label %output.plain
output.gate:
%output.gate.row = add i64 %row.base.wide, %gate.base.global
%output.gate.index = add i64 %output.gate.row, %output.local
%output.gate.index.phys = call i64 @recipe.window.index(i64 %output.gate.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%output.gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %output.gate.index.phys
%output.gate.value = load i8, ptr addrspace(1) %output.gate.ptr, align 1
%output.gate.factor = call i8 @recipe.sigmoid(i8 %output.gate.value)
%output.gated = call i8 @recipe.mul(i8 %attention, i8 %output.gate.factor)
br label %output.plain
output.plain:
%output.result = phi i8 [ %attention, %output.value.store ], [ %output.gated, %output.gate ]
store i8 %output.result, ptr addrspace(1) %output.ptr, align 1
%output.p.next = add i32 %output.p, %block
br label %output.loop
output.done:
call void @llvm.nvvm.barrier0()
br label %job.finish
job.finish:
%job.next = add i32 %job, %groups
br label %job.loop
exit:
ret void
}
define internal void @attention_cache_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) %kv.context,
i32 %rows, i32 %from, i32 %heads, i32 %kv.heads, i32 %value.heads, i32 %length, i32 %threads ) #3 { entry:
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%head.width = udiv i32 %from, %heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%value.channels = mul i32 %value.heads, %head.width
%value.plane = mul i32 %value.channels, %length
%kv.planes = add i32 %kv.plane, %value.plane
%row.stride = add i32 %from, %kv.planes
%total = mul i32 %rows, %kv.planes
%start = add i32 %lid, 0
br label %cache.loop
cache.loop:
%p = phi i32 [ %start, %entry ], [ %next, %cache.step ]
%more = icmp ult i32 %p, %total
br i1 %more, label %cache.step, label %cache.done
cache.step:
%row = udiv i32 %p, %kv.planes
%within = urem i32 %p, %kv.planes
%source.row = mul i32 %row, %row.stride
%source.index = add i32 %source.row, %from
%source.index.final = add i32 %source.index, %within
%source.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i32 %source.index.final
%value = load i8, ptr addrspace(1) %source.ptr, align 1
%kv.row = mul i32 %row, %kv.planes
%kv.index.final = add i32 %kv.row, %within
%kv.ptr = getelementptr inbounds i8, ptr addrspace(1) %kv.context, i32 %kv.index.final
%kv.encoded = call i8 @recipe.kv.encode(i8 %value)
store i8 %kv.encoded, ptr addrspace(1) %kv.ptr, align 1
%next = add i32 %p, %threads
br label %cache.loop
cache.done:
ret void
}
define internal void @attention_forward_matrix_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %weights,
ptr addrspace(1) nocapture writeonly %output, ptr addrspace(1) %context, ptr addrspace(1) %kv.context, i1 %carry,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, i8 %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, float %index.base ) #3 { entry:
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%width.double = call i8 @recipe.from.u32(i32 %head.width)
%scale.default = call i8 @recipe.sqrt(i8 %width.double)
%attention.zero = call i8 @recipe.from.u32(i32 0)
%attention.one = call i8 @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(i8 %attention.zero, i8 %epsilon)
%scale = select i1 %unscaled, i8 %attention.one, i8 %scale.default
br i1 %carry, label %attention.cache.entry, label %attention.cache.done
attention.cache.entry:
call void @attention_cache_body( ptr addrspace(1) %input, ptr addrspace(1) %kv.context, i32 %rows, i32 %from, i32 %heads, i32 %kv.heads, i32 %value.heads, i32 %length, i32 %threads )
call void @llvm.nvvm.barrier0()
br label %attention.cache.done
attention.cache.done:
%head.jobs = mul i32 %rows, %heads
%statistics.rows = mul i32 %head.jobs, %length
%rows.global = zext i32 %rows to i64 %from.global = zext i32 %from to i64 %channels.global = zext i32 %channels to i64 %heads.global = zext i32 %heads to i64 %length.global = zext i32 %length to i64 %head.width.global = zext i32 %head.width to i64
%head.jobs.global = mul i64 %rows.global, %heads.global %statistics.rows.global = mul i64 %head.jobs.global, %length.global
%input.row.stride.global = mul i64 %from.global, 3
%head.values = mul i32 %length, %head.width
%pair.values = mul i32 %length, %length
%q.base = add i32 0, 0
%k.base = add i32 %q.base, %head.values
%v.base = add i32 %k.base, %head.values
%p.base = add i32 %v.base, %head.values
%q.base.wide = zext i32 %q.base to i64 %k.base.wide = zext i32 %k.base to i64 %v.base.wide = zext i32 %v.base to i64 %p.base.wide = zext i32 %p.base to i64
br label %attention.forward.matrix.job.loop
attention.forward.matrix.job.loop:
%head.job = phi i32 [ %group, %attention.cache.done ], [ %head.job.next, %attention.forward.matrix.job.done ]
%head.job.more = icmp ult i32 %head.job, %head.jobs
br i1 %head.job.more, label %attention.forward.matrix.job.step, label %attention.forward.matrix.exit
attention.forward.matrix.job.step:
%head = urem i32 %head.job, %heads
%row = udiv i32 %head.job, %heads
%head.start = mul i32 %head, %head.width
%head.global = zext i32 %head to i64 %head.start.global = mul i64 %head.global, %head.width.global
%row.global = zext i32 %row to i64
%input.row.global = mul i64 %row.global, %input.row.stride.global
%output.row = mul i32 %row, %from %output.row.global = mul i64 %row.global, %from.global
br label %attention.forward.matrix.stage.channel.loop
attention.forward.matrix.stage.channel.loop:
%stage.channel.local = phi i32 [ %lid, %attention.forward.matrix.job.step ], [ %stage.channel.next, %attention.forward.matrix.stage.channel.done ]
%stage.channel.more = icmp ult i32 %stage.channel.local, %head.width
br i1 %stage.channel.more, label %attention.forward.matrix.stage.channel.step, label %attention.forward.matrix.stage.done
attention.forward.matrix.stage.channel.step:
%stage.channel = add i32 %head.start, %stage.channel.local
%stage.channel.wide = zext i32 %stage.channel to i64 %stage.channel.base = mul i64 %stage.channel.wide, %length.global
br label %attention.forward.matrix.stage.plane.loop
attention.forward.matrix.stage.plane.loop:
%stage.plane = phi i32 [ 0, %attention.forward.matrix.stage.channel.step ], [ %stage.plane.next, %attention.forward.matrix.stage.plane.done ]
%stage.plane.more = icmp ult i32 %stage.plane, 3
br i1 %stage.plane.more, label %attention.forward.matrix.stage.plane.step, label %attention.forward.matrix.stage.channel.done
attention.forward.matrix.stage.plane.step:
%stage.plane.wide = zext i32 %stage.plane to i64 %stage.input.plane = mul i64 %stage.plane.wide, %from.global
%stage.input.row = add i64 %input.row.global, %stage.input.plane
%stage.input.base = add i64 %stage.input.row, %stage.channel.base
%stage.shared.base = mul i32 %stage.plane, %head.values
br label %attention.forward.matrix.stage.position.loop
attention.forward.matrix.stage.position.loop:
%stage.position = phi i32 [ 0, %attention.forward.matrix.stage.plane.step ], [ %stage.position.next, %attention.forward.matrix.stage.position.advance ]
%stage.position.wide = zext i32 %stage.position to i64
%stage.position.more = icmp ult i32 %stage.position, %length
br i1 %stage.position.more, label %attention.forward.matrix.stage.vector.check, label %attention.forward.matrix.stage.plane.done
attention.forward.matrix.stage.vector.check:
%stage.position.remaining = sub i32 %length, %stage.position
%stage.vector = icmp uge i32 %stage.position.remaining, 16
br i1 %stage.vector, label %attention.forward.matrix.stage.vector, label %attention.forward.matrix.stage.scalar
attention.forward.matrix.stage.vector:
%stage.vector.index = add i64 %stage.input.base, %stage.position.wide
%stage.vector.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %stage.vector.index
%stage.vector.value = load <16 x i8>, ptr addrspace(1) %stage.vector.ptr, align 1
call void @contraction_stage_column16(<16 x i8> %stage.vector.value, i32 %stage.shared.base, i32 %stage.position, i32 %stage.channel.local, i32 %head.width)
%stage.vector.position.next = add i32 %stage.position, 16
br label %attention.forward.matrix.stage.position.advance
attention.forward.matrix.stage.scalar:
%stage.scalar.index = add i64 %stage.input.base, %stage.position.wide
%stage.scalar.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %stage.scalar.index
%stage.scalar.value = load i8, ptr addrspace(1) %stage.scalar.ptr, align 1
%stage.scalar.row = mul i32 %stage.position, %head.width
%stage.scalar.local = add i32 %stage.scalar.row, %stage.channel.local
%stage.scalar.index.shared = add i32 %stage.shared.base, %stage.scalar.local
%stage.scalar.ptr.shared = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %stage.scalar.index.shared
store i8 %stage.scalar.value, ptr addrspace(3) %stage.scalar.ptr.shared, align 1
%stage.scalar.position.next = add i32 %stage.position, 1
br label %attention.forward.matrix.stage.position.advance
attention.forward.matrix.stage.position.advance:
%stage.position.next = phi i32 [ %stage.vector.position.next, %attention.forward.matrix.stage.vector ], [ %stage.scalar.position.next, %attention.forward.matrix.stage.scalar ]
br label %attention.forward.matrix.stage.position.loop
attention.forward.matrix.stage.plane.done:
%stage.plane.next = add i32 %stage.plane, 1
br label %attention.forward.matrix.stage.plane.loop
attention.forward.matrix.stage.channel.done:
%stage.channel.next = add i32 %stage.channel.local, %block
br label %attention.forward.matrix.stage.channel.loop
attention.forward.matrix.stage.done:
call void @llvm.nvvm.barrier0()
%matrix.wave = udiv i32 %lid, 32
%matrix.lane = urem i32 %lid, 32
%matrix.waves = udiv i32 %block, 32
%matrix.lane.local = urem i32 %matrix.lane, 16
%matrix.lane.half = udiv i32 %matrix.lane, 16
%matrix.length.rounded = add i32 %length, 15
%matrix.tiles = udiv i32 %matrix.length.rounded, 16
%matrix.jobs = mul i32 %matrix.tiles, %matrix.tiles
br label %attention.forward.matrix.score.job.loop
attention.forward.matrix.score.job.loop:
%matrix.job = phi i32 [ %matrix.wave, %attention.forward.matrix.stage.done ], [ %matrix.job.next, %attention.forward.matrix.score.store.done ]
%matrix.job.more = icmp ult i32 %matrix.job, %matrix.jobs
br i1 %matrix.job.more, label %attention.forward.matrix.score.job.step, label %attention.forward.matrix.score.done
attention.forward.matrix.score.job.step:
%matrix.tile.q = udiv i32 %matrix.job, %matrix.tiles
%matrix.tile.k = urem i32 %matrix.job, %matrix.tiles
%matrix.q.tile = mul i32 %matrix.tile.q, 16
%matrix.k.tile = mul i32 %matrix.tile.k, 16
%matrix.q = add i32 %matrix.q.tile, %matrix.lane.local
%matrix.k = add i32 %matrix.k.tile, %matrix.lane.local
%matrix.q.valid = icmp ult i32 %matrix.q, %length
%matrix.k.valid = icmp ult i32 %matrix.k, %length
%matrix.q.safe = select i1 %matrix.q.valid, i32 %matrix.q, i32 0
%matrix.k.safe = select i1 %matrix.k.valid, i32 %matrix.k, i32 0
br label %attention.forward.matrix.score.width.loop
attention.forward.matrix.score.width.loop:
%matrix.width = phi i32 [ 0, %attention.forward.matrix.score.job.step ], [ %matrix.width.next, %attention.forward.matrix.score.width.step ]
%matrix.accumulator = phi <8 x float> [ zeroinitializer, %attention.forward.matrix.score.job.step ], [ %matrix.accumulator.next, %attention.forward.matrix.score.width.step ]
%matrix.width.more = icmp ult i32 %matrix.width, %head.width
br i1 %matrix.width.more, label %attention.forward.matrix.score.width.step, label %attention.forward.matrix.score.store.loop
attention.forward.matrix.score.width.step:
%matrix.q.row = mul i32 %matrix.q.safe, %head.width
%matrix.q.local = add i32 %matrix.q.row, %matrix.width
%matrix.q.index = add i32 %q.base, %matrix.q.local
%matrix.q.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.q.index
%matrix.q.fragment = load <16 x i8>, ptr addrspace(3) %matrix.q.ptr, align 2
%matrix.k.row = mul i32 %matrix.k.safe, %head.width
%matrix.k.local = add i32 %matrix.k.row, %matrix.width
%matrix.k.index = add i32 %k.base, %matrix.k.local
%matrix.k.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.k.index
%matrix.k.fragment = load <16 x i8>, ptr addrspace(3) %matrix.k.ptr, align 2
%matrix.accumulator.next = call <8 x float> @recipe.wmma(<16 x i8> %matrix.q.fragment, <16 x i8> %matrix.k.fragment, <8 x float> %matrix.accumulator)
%matrix.width.next = add i32 %matrix.width, 16
br label %attention.forward.matrix.score.width.loop
attention.forward.matrix.score.store.loop:
%matrix.output = phi i32 [ 0, %attention.forward.matrix.score.width.loop ], [ %matrix.output.next, %attention.forward.matrix.score.store.next ]
%matrix.output.more = icmp ult i32 %matrix.output, 8
br i1 %matrix.output.more, label %attention.forward.matrix.score.store.check, label %attention.forward.matrix.score.store.done
attention.forward.matrix.score.store.check:
%matrix.output.twice = mul i32 %matrix.output, 2
%matrix.query.local = add i32 %matrix.output.twice, %matrix.lane.half
%matrix.query = add i32 %matrix.q.tile, %matrix.query.local
%matrix.query.valid = icmp ult i32 %matrix.query, %length
%matrix.pair.valid = and i1 %matrix.query.valid, %matrix.k.valid
br i1 %matrix.pair.valid, label %attention.forward.matrix.score.store, label %attention.forward.matrix.score.store.next
attention.forward.matrix.score.store:
%matrix.score.wide = extractelement <8 x float> %matrix.accumulator, i32 %matrix.output
%matrix.score.raw = call i8 @recipe.encode(float %matrix.score.wide)
%matrix.score.scaled = call i8 @recipe.div(i8 %matrix.score.raw, i8 %scale)
%matrix.causal = icmp ule i32 %matrix.k, %matrix.query
%matrix.score = select i1 %matrix.causal, i8 %matrix.score.scaled, i8 254
%matrix.pair.row = mul i32 %matrix.query, %length
%matrix.pair.local = add i32 %matrix.pair.row, %matrix.k
%matrix.p.index = add i32 %p.base, %matrix.pair.local
%matrix.p.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.p.index
store i8 %matrix.score, ptr addrspace(3) %matrix.p.ptr, align 1
br label %attention.forward.matrix.score.store.next
attention.forward.matrix.score.store.next:
%matrix.output.next = add i32 %matrix.output, 1
br label %attention.forward.matrix.score.store.loop
attention.forward.matrix.score.store.done:
%matrix.job.next = add i32 %matrix.job, %matrix.waves
br label %attention.forward.matrix.score.job.loop
attention.forward.matrix.score.done:
call void @llvm.nvvm.barrier0()
br label %attention.forward.matrix.softmax.loop
attention.forward.matrix.softmax.loop:
%softmax.query = phi i32 [ %lid, %attention.forward.matrix.score.done ], [ %softmax.query.next, %attention.forward.matrix.softmax.store ]
%softmax.more = icmp ult i32 %softmax.query, %length
br i1 %softmax.more, label %attention.forward.matrix.maximum.loop, label %attention.forward.matrix.softmax.done
attention.forward.matrix.maximum.loop:
%maximum.key = phi i32 [ 0, %attention.forward.matrix.softmax.loop ], [ %maximum.key.next, %attention.forward.matrix.maximum.step ]
%maximum = phi i8 [ 254, %attention.forward.matrix.softmax.loop ], [ %maximum.next, %attention.forward.matrix.maximum.step ]
%maximum.more = icmp ult i32 %maximum.key, %length
br i1 %maximum.more, label %attention.forward.matrix.maximum.step, label %attention.forward.matrix.probability.loop
attention.forward.matrix.maximum.step:
%maximum.row = mul i32 %softmax.query, %length
%maximum.local = add i32 %maximum.row, %maximum.key
%maximum.index = add i32 %p.base, %maximum.local
%maximum.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %maximum.index
%maximum.score = load i8, ptr addrspace(3) %maximum.ptr, align 1
%maximum.larger = call i1 @recipe.ogt(i8 %maximum.score, i8 %maximum)
%maximum.next = select i1 %maximum.larger, i8 %maximum.score, i8 %maximum
%maximum.key.next = add i32 %maximum.key, 1
br label %attention.forward.matrix.maximum.loop
attention.forward.matrix.probability.loop:
%probability.key = phi i32 [ 0, %attention.forward.matrix.maximum.loop ], [ %probability.key.next, %attention.forward.matrix.probability.step ]
%denominator = phi i8 [ 0, %attention.forward.matrix.maximum.loop ], [ %denominator.next, %attention.forward.matrix.probability.step ]
%probability.more = icmp ult i32 %probability.key, %length
br i1 %probability.more, label %attention.forward.matrix.probability.step, label %attention.forward.matrix.normalize.loop
attention.forward.matrix.probability.step:
%probability.row = mul i32 %softmax.query, %length
%probability.local = add i32 %probability.row, %probability.key
%probability.index = add i32 %p.base, %probability.local
%probability.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %probability.index
%probability.score = load i8, ptr addrspace(3) %probability.ptr, align 1
%probability.centered = call i8 @recipe.sub(i8 %probability.score, i8 %maximum)
%probability.value = call i8 @recipe.exp(i8 %probability.centered)
store i8 %probability.value, ptr addrspace(3) %probability.ptr, align 1
%denominator.next = call i8 @recipe.add(i8 %denominator, i8 %probability.value)
%probability.key.next = add i32 %probability.key, 1
br label %attention.forward.matrix.probability.loop
attention.forward.matrix.normalize.loop:
%normalize.key = phi i32 [ 0, %attention.forward.matrix.probability.loop ], [ %normalize.key.next, %attention.forward.matrix.normalize.step ]
%normalize.more = icmp ult i32 %normalize.key, %length
br i1 %normalize.more, label %attention.forward.matrix.normalize.step, label %attention.forward.matrix.softmax.store
attention.forward.matrix.normalize.step:
%normalize.row = mul i32 %softmax.query, %length
%normalize.local = add i32 %normalize.row, %normalize.key
%normalize.index = add i32 %p.base, %normalize.local
%normalize.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %normalize.index
%normalize.value = load i8, ptr addrspace(3) %normalize.ptr, align 1
%normalized = call i8 @recipe.div(i8 %normalize.value, i8 %denominator)
store i8 %normalized, ptr addrspace(3) %normalize.ptr, align 1
%normalize.key.next = add i32 %normalize.key, 1
br label %attention.forward.matrix.normalize.loop
attention.forward.matrix.softmax.store:
%head.job.wide = zext i32 %head.job to i64 %softmax.query.wide = zext i32 %softmax.query to i64 %statistics.base = mul i64 %head.job.wide, %length.global
%statistics.index = add i64 %statistics.base, %softmax.query.wide
%maximum.context.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %statistics.index
store i8 %maximum, ptr addrspace(1) %maximum.context.ptr, align 1
%denominator.index = add i64 %statistics.rows.global, %statistics.index
%denominator.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %denominator.index
store i8 %denominator, ptr addrspace(1) %denominator.ptr, align 1
%softmax.query.next = add i32 %softmax.query, %block
br label %attention.forward.matrix.softmax.loop
attention.forward.matrix.softmax.done:
call void @llvm.nvvm.barrier0()
call void @attention_matrix_product(ptr addrspace(1) %output, i32 3, i64 %p.base.wide, i64 %v.base.wide, i64 %output.row.global, i32 %from, i32 %head.start, i32 %length, i32 %head.width, i8 %scale, i32 %lid, i32 %block)
call void @llvm.nvvm.barrier0()
br label %attention.forward.matrix.job.done
attention.forward.matrix.job.done:
%head.job.next = add i32 %head.job, %groups
br label %attention.forward.matrix.job.loop
attention.forward.matrix.exit:
ret void
}
define internal void @attention_matrix_product(
ptr addrspace(1) %previous, i32 %mode, i64 %left.base, i64 %right.base,
i64 %row.base, i32 %from, i32 %head.start, i32 %length, i32 %head.width,
i8 %scale, i32 %lid, i32 %block ) #1 { entry:
%dq = icmp eq i32 %mode, 0
%forward = icmp eq i32 %mode, 3
%direct = or i1 %dq, %forward
%dv = icmp eq i32 %mode, 2
%unscaled = or i1 %dv, %forward
%wave = udiv i32 %lid, 32
%lane = urem i32 %lid, 32
%waves = udiv i32 %block, 32
%lane.local = urem i32 %lane, 16
%lane.half = udiv i32 %lane, 16
%m.rounded = add i32 %length, 15
%m.tiles = udiv i32 %m.rounded, 16
%n.rounded = add i32 %head.width, 15
%n.tiles = udiv i32 %n.rounded, 16
%jobs = mul i32 %m.tiles, %n.tiles
%length.wide = zext i32 %length to i64
br label %attention.matrix.gradient.job.loop
attention.matrix.gradient.job.loop:
%job = phi i32 [ %wave, %entry ], [ %job.next, %attention.matrix.gradient.store.done ]
%job.more = icmp ult i32 %job, %jobs
br i1 %job.more, label %attention.matrix.gradient.job.step, label %attention.matrix.gradient.exit
attention.matrix.gradient.job.step:
%tile.m = udiv i32 %job, %n.tiles
%tile.n = urem i32 %job, %n.tiles
%m.tile = mul i32 %tile.m, 16
%n.tile = mul i32 %tile.n, 16
%m = add i32 %m.tile, %lane.local
%n = add i32 %n.tile, %lane.local
%m.valid = icmp ult i32 %m, %length
%n.valid = icmp ult i32 %n, %head.width
%m.safe = select i1 %m.valid, i32 %m, i32 0
%n.safe = select i1 %n.valid, i32 %n, i32 0
br label %attention.matrix.gradient.k.loop
attention.matrix.gradient.k.loop:
%k.base = phi i32 [ 0, %attention.matrix.gradient.job.step ], [ %k.next, %attention.matrix.gradient.fragment.done ]
%accumulator = phi <8 x float> [ zeroinitializer, %attention.matrix.gradient.job.step ], [ %accumulator.next, %attention.matrix.gradient.fragment.done ]
%k.more = icmp ult i32 %k.base, %length
br i1 %k.more, label %attention.matrix.gradient.fragment.loop, label %attention.matrix.gradient.store.loop
attention.matrix.gradient.fragment.loop:
%fragment = phi i32 [ 0, %attention.matrix.gradient.k.loop ], [ %fragment.next, %attention.matrix.gradient.fragment.step ]
%left.fragment = phi <16 x i8> [ zeroinitializer, %attention.matrix.gradient.k.loop ], [ %left.fragment.next, %attention.matrix.gradient.fragment.step ]
%right.fragment = phi <16 x i8> [ zeroinitializer, %attention.matrix.gradient.k.loop ], [ %right.fragment.next, %attention.matrix.gradient.fragment.step ]
%fragment.more = icmp ult i32 %fragment, 16
br i1 %fragment.more, label %attention.matrix.gradient.fragment.step, label %attention.matrix.gradient.fragment.done
attention.matrix.gradient.fragment.step:
%term = add i32 %k.base, %fragment
%term.valid = icmp ult i32 %term, %length
%term.safe = select i1 %term.valid, i32 %term, i32 0
%m.safe.wide = zext i32 %m.safe to i64 %term.safe.wide = zext i32 %term.safe to i64
%left.direct.row = mul i64 %m.safe.wide, %length.wide
%left.direct.local = add i64 %left.direct.row, %term.safe.wide
%left.transpose.row = mul i64 %term.safe.wide, %length.wide
%left.transpose.local = add i64 %left.transpose.row, %m.safe.wide
%left.local = select i1 %direct, i64 %left.direct.local, i64 %left.transpose.local
%left.index = add i64 %left.base, %left.local
%left.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i64 0, i64 %left.index
%left.loaded = load i8, ptr addrspace(3) %left.ptr, align 1
%head.width.wide = zext i32 %head.width to i64 %n.safe.wide = zext i32 %n.safe to i64 %right.row = mul i64 %term.safe.wide, %head.width.wide
%right.local = add i64 %right.row, %n.safe.wide
%right.index = add i64 %right.base, %right.local
%right.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i64 0, i64 %right.index
%right.loaded = load i8, ptr addrspace(3) %right.ptr, align 1
%left.valid = and i1 %m.valid, %term.valid
%right.valid = and i1 %n.valid, %term.valid
%left.value = select i1 %left.valid, i8 %left.loaded, i8 0
%right.value = select i1 %right.valid, i8 %right.loaded, i8 0
%left.fragment.next = insertelement <16 x i8> %left.fragment, i8 %left.value, i32 %fragment
%right.fragment.next = insertelement <16 x i8> %right.fragment, i8 %right.value, i32 %fragment
%fragment.next = add i32 %fragment, 1
br label %attention.matrix.gradient.fragment.loop
attention.matrix.gradient.fragment.done:
%accumulator.next = call <8 x float> @recipe.wmma(<16 x i8> %left.fragment, <16 x i8> %right.fragment, <8 x float> %accumulator)
%k.next = add i32 %k.base, 16
br label %attention.matrix.gradient.k.loop
attention.matrix.gradient.store.loop:
%output = phi i32 [ 0, %attention.matrix.gradient.k.loop ], [ %output.next, %attention.matrix.gradient.store.next ]
%output.more = icmp ult i32 %output, 8
br i1 %output.more, label %attention.matrix.gradient.store.check, label %attention.matrix.gradient.store.done
attention.matrix.gradient.store.check:
%output.twice = mul i32 %output, 2
%output.m.local = add i32 %output.twice, %lane.half
%output.m = add i32 %m.tile, %output.m.local
%output.m.valid = icmp ult i32 %output.m, %length
%output.valid = and i1 %output.m.valid, %n.valid
br i1 %output.valid, label %attention.matrix.gradient.store, label %attention.matrix.gradient.store.next
attention.matrix.gradient.store:
%output.wide = extractelement <8 x float> %accumulator, i32 %output
%output.raw = call i8 @recipe.encode(float %output.wide)
%output.scaled = call i8 @recipe.div(i8 %output.raw, i8 %scale)
%output.value = select i1 %unscaled, i8 %output.raw, i8 %output.scaled
%mode.wide = zext i32 %mode to i64 %from.wide = zext i32 %from to i64 %output.plane.raw = mul i64 %mode.wide, %from.wide
%output.plane = select i1 %forward, i64 0, i64 %output.plane.raw
%output.row = add i64 %row.base, %output.plane
%head.start.wide = zext i32 %head.start to i64 %n.wide = zext i32 %n to i64 %output.channel = add i64 %head.start.wide, %n.wide
%output.channel.base = mul i64 %output.channel, %length.wide
%output.m.wide = zext i32 %output.m to i64 %output.local = add i64 %output.channel.base, %output.m.wide
%output.index = add i64 %output.row, %output.local
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %previous, i64 %output.index
store i8 %output.value, ptr addrspace(1) %output.ptr, align 1
br label %attention.matrix.gradient.store.next
attention.matrix.gradient.store.next:
%output.next = add i32 %output, 1
br label %attention.matrix.gradient.store.loop
attention.matrix.gradient.store.done:
%job.next = add i32 %job, %waves
br label %attention.matrix.gradient.job.loop
attention.matrix.gradient.exit:
ret void
}
define internal void @attention_reverse_matrix_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %output, ptr addrspace(1) %context,
ptr addrspace(1) nocapture readonly %delta, ptr addrspace(1) nocapture writeonly %previous,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, i8 %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, float %index.base ) #3 { entry:
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%width.double = call i8 @recipe.from.u32(i32 %head.width)
%scale.default = call i8 @recipe.sqrt(i8 %width.double)
%attention.zero = call i8 @recipe.from.u32(i32 0)
%attention.one = call i8 @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(i8 %attention.zero, i8 %epsilon)
%scale = select i1 %unscaled, i8 %attention.one, i8 %scale.default
%head.jobs = mul i32 %rows, %heads
%statistics.rows = mul i32 %head.jobs, %length
%head.values = mul i32 %length, %head.width
%pair.values = mul i32 %length, %length
%rows.global = zext i32 %rows to i64 %from.global = zext i32 %from to i64 %channels.global = zext i32 %channels to i64 %heads.global = zext i32 %heads to i64 %length.global = zext i32 %length to i64 %head.width.global = zext i32 %head.width to i64
%head.jobs.global = mul i64 %rows.global, %heads.global %statistics.rows.global = mul i64 %head.jobs.global, %length.global %head.values.global = mul i64 %length.global, %head.width.global %pair.values.global = mul i64 %length.global, %length.global
%q.base = add i32 0, 0
%k.base = add i32 %q.base, %head.values
%v.base = add i32 %k.base, %head.values
%do.base = add i32 %v.base, %head.values
%p.base = add i32 %do.base, %head.values
%ds.base = add i32 %p.base, %pair.values
%d.base = add i32 %ds.base, %pair.values
%q.base.global = add i64 0, 0 %k.base.global = add i64 %q.base.global, %head.values.global %v.base.global = add i64 %k.base.global, %head.values.global %do.base.global = add i64 %v.base.global, %head.values.global %p.base.global = add i64 %do.base.global, %head.values.global %ds.base.global = add i64 %p.base.global, %pair.values.global %d.base.global = add i64 %ds.base.global, %pair.values.global
br label %attention.matrix.job.loop
attention.matrix.job.loop:
%head.job = phi i32 [ %group, %entry ], [ %head.job.next, %attention.matrix.job.done ]
%head.job.more = icmp ult i32 %head.job, %head.jobs
br i1 %head.job.more, label %attention.matrix.job.step, label %attention.matrix.exit
attention.matrix.job.step:
%head = urem i32 %head.job, %heads
%row = udiv i32 %head.job, %heads
%head.start = mul i32 %head, %head.width
%head.job.global = zext i32 %head.job to i64 %row.global = zext i32 %row to i64 %head.start.global = zext i32 %head.start to i64
%input.row.stride = mul i64 %from.global, 3
%input.row = mul i64 %row.global, %input.row.stride
%output.row = mul i64 %row.global, %from.global
br label %attention.matrix.stage.channel.loop
attention.matrix.stage.channel.loop:
%stage.channel.local = phi i32 [ %lid, %attention.matrix.job.step ], [ %stage.channel.next, %attention.matrix.stage.channel.done ]
%stage.channel.more = icmp ult i32 %stage.channel.local, %head.width
br i1 %stage.channel.more, label %attention.matrix.stage.channel.step, label %attention.matrix.stage.done
attention.matrix.stage.channel.step:
%stage.channel = add i32 %head.start, %stage.channel.local
%stage.channel.global = zext i32 %stage.channel to i64 %stage.channel.base = mul i64 %stage.channel.global, %length.global
br label %attention.matrix.stage.plane.loop
attention.matrix.stage.plane.loop:
%stage.plane = phi i32 [ 0, %attention.matrix.stage.channel.step ], [ %stage.plane.next, %attention.matrix.stage.plane.done ]
%stage.plane.more = icmp ult i32 %stage.plane, 4
br i1 %stage.plane.more, label %attention.matrix.stage.plane.step, label %attention.matrix.stage.channel.done
attention.matrix.stage.plane.step:
%stage.plane.global = zext i32 %stage.plane to i64 %stage.input.plane = mul i64 %stage.plane.global, %from.global
%stage.input.row = add i64 %input.row, %stage.input.plane
%stage.input.base = add i64 %stage.input.row, %stage.channel.base
%stage.delta.base = add i64 %output.row, %stage.channel.base
%stage.shared.base = mul i32 %stage.plane, %head.values
%stage.is.delta = icmp eq i32 %stage.plane, 3
br label %attention.matrix.stage.position.loop
attention.matrix.stage.position.loop:
%stage.position = phi i32 [ 0, %attention.matrix.stage.plane.step ], [ %stage.position.next, %attention.matrix.stage.position.advance ]
%stage.position.global = zext i32 %stage.position to i64
%stage.position.more = icmp ult i32 %stage.position, %length
br i1 %stage.position.more, label %attention.matrix.stage.vector.check, label %attention.matrix.stage.plane.done
attention.matrix.stage.vector.check:
%stage.position.remaining = sub i32 %length, %stage.position
%stage.vector = icmp uge i32 %stage.position.remaining, 16
br i1 %stage.vector, label %attention.matrix.stage.vector.select, label %attention.matrix.stage.scalar.select
attention.matrix.stage.vector.select:
br i1 %stage.is.delta, label %attention.matrix.stage.vector.delta, label %attention.matrix.stage.vector.input
attention.matrix.stage.vector.input:
%stage.vector.input.index = add i64 %stage.input.base, %stage.position.global
%stage.vector.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %stage.vector.input.index
%stage.vector.input.value = load <16 x i8>, ptr addrspace(1) %stage.vector.input.ptr, align 1
br label %attention.matrix.stage.vector.store
attention.matrix.stage.vector.delta:
%stage.vector.delta.index = add i64 %stage.delta.base, %stage.position.global
%stage.vector.delta.ptr = getelementptr inbounds i8, ptr addrspace(1) %delta, i64 %stage.vector.delta.index
%stage.vector.delta.value = load <16 x i8>, ptr addrspace(1) %stage.vector.delta.ptr, align 1
br label %attention.matrix.stage.vector.store
attention.matrix.stage.vector.store:
%stage.vector.value = phi <16 x i8> [ %stage.vector.input.value, %attention.matrix.stage.vector.input ], [ %stage.vector.delta.value, %attention.matrix.stage.vector.delta ]
call void @contraction_stage_column16(<16 x i8> %stage.vector.value, i32 %stage.shared.base, i32 %stage.position, i32 %stage.channel.local, i32 %head.width)
%stage.vector.position.next = add i32 %stage.position, 16
br label %attention.matrix.stage.position.advance
attention.matrix.stage.scalar.select:
br i1 %stage.is.delta, label %attention.matrix.stage.scalar.delta, label %attention.matrix.stage.scalar.input
attention.matrix.stage.scalar.input:
%stage.scalar.input.index = add i64 %stage.input.base, %stage.position.global
%stage.scalar.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %stage.scalar.input.index
%stage.scalar.input.value = load i8, ptr addrspace(1) %stage.scalar.input.ptr, align 1
br label %attention.matrix.stage.scalar.store
attention.matrix.stage.scalar.delta:
%stage.scalar.delta.index = add i64 %stage.delta.base, %stage.position.global
%stage.scalar.delta.ptr = getelementptr inbounds i8, ptr addrspace(1) %delta, i64 %stage.scalar.delta.index
%stage.scalar.delta.value = load i8, ptr addrspace(1) %stage.scalar.delta.ptr, align 1
br label %attention.matrix.stage.scalar.store
attention.matrix.stage.scalar.store:
%stage.scalar.value = phi i8 [ %stage.scalar.input.value, %attention.matrix.stage.scalar.input ], [ %stage.scalar.delta.value, %attention.matrix.stage.scalar.delta ]
%stage.scalar.row = mul i32 %stage.position, %head.width
%stage.scalar.local = add i32 %stage.scalar.row, %stage.channel.local
%stage.scalar.index.shared = add i32 %stage.shared.base, %stage.scalar.local
%stage.scalar.ptr.shared = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %stage.scalar.index.shared
store i8 %stage.scalar.value, ptr addrspace(3) %stage.scalar.ptr.shared, align 1
%stage.scalar.position.next = add i32 %stage.position, 1
br label %attention.matrix.stage.position.advance
attention.matrix.stage.position.advance:
%stage.position.next = phi i32 [ %stage.vector.position.next, %attention.matrix.stage.vector.store ], [ %stage.scalar.position.next, %attention.matrix.stage.scalar.store ]
br label %attention.matrix.stage.position.loop
attention.matrix.stage.plane.done:
%stage.plane.next = add i32 %stage.plane, 1
br label %attention.matrix.stage.plane.loop
attention.matrix.stage.channel.done:
%stage.channel.next = add i32 %stage.channel.local, %block
br label %attention.matrix.stage.channel.loop
attention.matrix.stage.done:
call void @llvm.nvvm.barrier0()
br label %attention.matrix.d.loop
attention.matrix.d.loop:
%d.query = phi i32 [ %lid, %attention.matrix.stage.done ], [ %d.query.next, %attention.matrix.d.store ]
%d.more = icmp ult i32 %d.query, %length
br i1 %d.more, label %attention.matrix.d.sum.loop, label %attention.matrix.d.done
attention.matrix.d.sum.loop:
%d.channel = phi i32 [ 0, %attention.matrix.d.loop ], [ %d.channel.next, %attention.matrix.d.sum.step ]
%d.sum = phi i8 [ 0, %attention.matrix.d.loop ], [ %d.sum.next, %attention.matrix.d.sum.step ]
%d.channel.more = icmp ult i32 %d.channel, %head.width
br i1 %d.channel.more, label %attention.matrix.d.sum.step, label %attention.matrix.d.store
attention.matrix.d.sum.step:
%d.query.wide = zext i32 %d.query to i64 %d.channel.wide = zext i32 %d.channel to i64 %d.shared.row = mul i64 %d.query.wide, %head.width.global
%d.shared.local = add i64 %d.shared.row, %d.channel.wide
%d.do.index = add i64 %do.base.global, %d.shared.local
%d.do.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i64 0, i64 %d.do.index
%d.do = load i8, ptr addrspace(3) %d.do.ptr, align 1
%d.output.channel = add i64 %head.start.global, %d.channel.wide
%d.output.channel.base = mul i64 %d.output.channel, %length.global
%d.output.local = add i64 %d.output.channel.base, %d.query.wide
%d.output.index = add i64 %output.row, %d.output.local
%d.output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %d.output.index
%d.output = load i8, ptr addrspace(1) %d.output.ptr, align 1
%d.term = call i8 @recipe.mul(i8 %d.do, i8 %d.output)
%d.sum.next = call i8 @recipe.add(i8 %d.sum, i8 %d.term)
%d.channel.next = add i32 %d.channel, 1
br label %attention.matrix.d.sum.loop
attention.matrix.d.store:
%d.query.wide.store = zext i32 %d.query to i64 %d.index = add i64 %d.base.global, %d.query.wide.store
%d.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i64 0, i64 %d.index
store i8 %d.sum, ptr addrspace(3) %d.ptr, align 1
%d.query.next = add i32 %d.query, %block
br label %attention.matrix.d.loop
attention.matrix.d.done:
call void @llvm.nvvm.barrier0()
%matrix.wave = udiv i32 %lid, 32
%matrix.lane = urem i32 %lid, 32
%matrix.waves = udiv i32 %block, 32
%matrix.lane.local = urem i32 %matrix.lane, 16
%matrix.lane.half = udiv i32 %matrix.lane, 16
%matrix.length.rounded = add i32 %length, 15
%matrix.tiles = udiv i32 %matrix.length.rounded, 16
%matrix.jobs = mul i32 %matrix.tiles, %matrix.tiles
br label %attention.matrix.score.job.loop
attention.matrix.score.job.loop:
%matrix.job = phi i32 [ %matrix.wave, %attention.matrix.d.done ], [ %matrix.job.next, %attention.matrix.score.store.done ]
%matrix.job.more = icmp ult i32 %matrix.job, %matrix.jobs
br i1 %matrix.job.more, label %attention.matrix.score.job.step, label %attention.matrix.score.done
attention.matrix.score.job.step:
%matrix.tile.q = udiv i32 %matrix.job, %matrix.tiles
%matrix.tile.k = urem i32 %matrix.job, %matrix.tiles
%matrix.q.tile = mul i32 %matrix.tile.q, 16
%matrix.k.tile = mul i32 %matrix.tile.k, 16
%matrix.q = add i32 %matrix.q.tile, %matrix.lane.local
%matrix.k = add i32 %matrix.k.tile, %matrix.lane.local
%matrix.q.valid = icmp ult i32 %matrix.q, %length
%matrix.k.valid = icmp ult i32 %matrix.k, %length
%matrix.q.safe = select i1 %matrix.q.valid, i32 %matrix.q, i32 0
%matrix.k.safe = select i1 %matrix.k.valid, i32 %matrix.k, i32 0
br label %attention.matrix.score.width.loop
attention.matrix.score.width.loop:
%matrix.width = phi i32 [ 0, %attention.matrix.score.job.step ], [ %matrix.width.next, %attention.matrix.score.width.step ]
%matrix.score.accumulator = phi <8 x float> [ zeroinitializer, %attention.matrix.score.job.step ], [ %matrix.score.accumulator.next, %attention.matrix.score.width.step ]
%matrix.dp.accumulator = phi <8 x float> [ zeroinitializer, %attention.matrix.score.job.step ], [ %matrix.dp.accumulator.next, %attention.matrix.score.width.step ]
%matrix.width.more = icmp ult i32 %matrix.width, %head.width
br i1 %matrix.width.more, label %attention.matrix.score.width.step, label %attention.matrix.score.store.loop
attention.matrix.score.width.step:
%matrix.q.row = mul i32 %matrix.q.safe, %head.width
%matrix.q.local = add i32 %matrix.q.row, %matrix.width
%matrix.q.index = add i32 %q.base, %matrix.q.local
%matrix.q.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.q.index
%matrix.q.fragment = load <16 x i8>, ptr addrspace(3) %matrix.q.ptr, align 2
%matrix.k.row = mul i32 %matrix.k.safe, %head.width
%matrix.k.local = add i32 %matrix.k.row, %matrix.width
%matrix.k.index = add i32 %k.base, %matrix.k.local
%matrix.k.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.k.index
%matrix.k.fragment = load <16 x i8>, ptr addrspace(3) %matrix.k.ptr, align 2
%matrix.do.index = add i32 %do.base, %matrix.q.local
%matrix.do.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.do.index
%matrix.do.fragment = load <16 x i8>, ptr addrspace(3) %matrix.do.ptr, align 2
%matrix.v.index = add i32 %v.base, %matrix.k.local
%matrix.v.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.v.index
%matrix.v.fragment = load <16 x i8>, ptr addrspace(3) %matrix.v.ptr, align 2
%matrix.score.accumulator.next = call <8 x float> @recipe.wmma(<16 x i8> %matrix.q.fragment, <16 x i8> %matrix.k.fragment, <8 x float> %matrix.score.accumulator)
%matrix.dp.accumulator.next = call <8 x float> @recipe.wmma(<16 x i8> %matrix.do.fragment, <16 x i8> %matrix.v.fragment, <8 x float> %matrix.dp.accumulator)
%matrix.width.next = add i32 %matrix.width, 16
br label %attention.matrix.score.width.loop
attention.matrix.score.store.loop:
%matrix.output = phi i32 [ 0, %attention.matrix.score.width.loop ], [ %matrix.output.next, %attention.matrix.score.store.next ]
%matrix.output.more = icmp ult i32 %matrix.output, 8
br i1 %matrix.output.more, label %attention.matrix.score.store.check, label %attention.matrix.score.store.done
attention.matrix.score.store.check:
%matrix.output.twice = mul i32 %matrix.output, 2
%matrix.query.local = add i32 %matrix.output.twice, %matrix.lane.half
%matrix.query = add i32 %matrix.q.tile, %matrix.query.local
%matrix.query.valid = icmp ult i32 %matrix.query, %length
%matrix.pair.valid = and i1 %matrix.query.valid, %matrix.k.valid
br i1 %matrix.pair.valid, label %attention.matrix.score.complete, label %attention.matrix.score.store.next
attention.matrix.score.complete:
%matrix.score.wide = extractelement <8 x float> %matrix.score.accumulator, i32 %matrix.output
%matrix.score.raw = call i8 @recipe.encode(float %matrix.score.wide)
%matrix.score = call i8 @recipe.div(i8 %matrix.score.raw, i8 %scale)
%matrix.dp.wide = extractelement <8 x float> %matrix.dp.accumulator, i32 %matrix.output
%matrix.dp = call i8 @recipe.encode(float %matrix.dp.wide)
%matrix.causal = icmp ule i32 %matrix.k, %matrix.query
%matrix.head.job.global = zext i32 %head.job to i64
%matrix.query.global = zext i32 %matrix.query to i64
%matrix.statistics.base = mul i64 %matrix.head.job.global, %length.global
%matrix.statistics.index = add i64 %matrix.statistics.base, %matrix.query.global
%matrix.maximum.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %matrix.statistics.index
%matrix.maximum = load i8, ptr addrspace(1) %matrix.maximum.ptr, align 1
%matrix.denominator.index = add i64 %statistics.rows.global, %matrix.statistics.index
%matrix.denominator.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %matrix.denominator.index
%matrix.denominator = load i8, ptr addrspace(1) %matrix.denominator.ptr, align 1
%matrix.centered = call i8 @recipe.sub(i8 %matrix.score, i8 %matrix.maximum)
%matrix.exponential = call i8 @recipe.exp(i8 %matrix.centered)
%matrix.probability.raw = call i8 @recipe.div(i8 %matrix.exponential, i8 %matrix.denominator)
%matrix.probability = select i1 %matrix.causal, i8 %matrix.probability.raw, i8 0
%matrix.d.index = add i32 %d.base, %matrix.query
%matrix.d.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.d.index
%matrix.d = load i8, ptr addrspace(3) %matrix.d.ptr, align 1
%matrix.dp.centered = call i8 @recipe.sub(i8 %matrix.dp, i8 %matrix.d)
%matrix.derivative = call i8 @recipe.mul(i8 %matrix.probability, i8 %matrix.dp.centered)
%matrix.pair.row = mul i32 %matrix.query, %length
%matrix.pair.local = add i32 %matrix.pair.row, %matrix.k
%matrix.p.index = add i32 %p.base, %matrix.pair.local
%matrix.p.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.p.index
store i8 %matrix.probability, ptr addrspace(3) %matrix.p.ptr, align 1
%matrix.ds.index = add i32 %ds.base, %matrix.pair.local
%matrix.ds.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %matrix.ds.index
store i8 %matrix.derivative, ptr addrspace(3) %matrix.ds.ptr, align 1
br label %attention.matrix.score.store.next
attention.matrix.score.store.next:
%matrix.output.next = add i32 %matrix.output, 1
br label %attention.matrix.score.store.loop
attention.matrix.score.store.done:
%matrix.job.next = add i32 %matrix.job, %matrix.waves
br label %attention.matrix.score.job.loop
attention.matrix.score.done:
call void @llvm.nvvm.barrier0()
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 0, i64 %ds.base.global, i64 %k.base.global, i64 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, i8 %scale, i32 %lid, i32 %block)
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 1, i64 %ds.base.global, i64 %q.base.global, i64 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, i8 %scale, i32 %lid, i32 %block)
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 2, i64 %p.base.global, i64 %do.base.global, i64 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, i8 %scale, i32 %lid, i32 %block)
call void @llvm.nvvm.barrier0()
br label %attention.matrix.job.done
attention.matrix.job.done:
%head.job.next = add i32 %head.job, %groups
br label %attention.matrix.job.loop
attention.matrix.exit:
ret void
}
define internal void @attention_reverse_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %output, ptr addrspace(1) %context,
ptr addrspace(1) nocapture readonly %delta, ptr addrspace(1) nocapture writeonly %previous,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, i8 %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, float %index.base ) #3 { entry:
%state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true)
%lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%head.width.double = call i8 @recipe.from.u32(i32 %head.width)
%scale.default = call i8 @recipe.sqrt(i8 %head.width.double)
%attention.zero = call i8 @recipe.from.u32(i32 0)
%attention.one = call i8 @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(i8 %attention.zero, i8 %epsilon)
%scale = select i1 %unscaled, i8 %attention.one, i8 %scale.default
%scale.wide = call float @recipe.decode(i8 %scale)
%kv.group = udiv i32 %heads, %kv.heads
%value.group = udiv i32 %heads, %value.heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%value.channels = mul i32 %value.heads, %head.width
%value.plane = mul i32 %value.channels, %length
%kv.planes = add i32 %kv.plane, %value.plane
%value.plane.base = add i32 %from, %kv.plane
%index.query.channels = mul i32 %index.heads, %index.width
%index.channels = add i32 %index.query.channels, %index.width
%index.plane = mul i32 %index.channels, %length
%gate.plane = select i1 %gate, i32 %from, i32 0
%index.query.base = add i32 %from, %kv.planes
%gate.base = add i32 %index.query.base, 0
%row.stride = add i32 %gate.base, %gate.plane
%select = icmp ne i32 %select.block, 0
%block.divisor = select i1 %select, i32 %select.block, i32 1
%blocks.numerator = add i32 %length, %block.divisor
%blocks.less = sub i32 %blocks.numerator, 1
%blocks.full = udiv i32 %blocks.less, %block.divisor
%blocks = select i1 %select, i32 %blocks.full, i32 0
%score.stride = mul i32 %blocks, 2
%head.jobs = mul i32 %rows, %heads
%statistics.rows = mul i32 %head.jobs, %length
%statistics.denominator.base = add i32 0, %statistics.rows
%representative.base = mul i32 %statistics.rows, 2
%representative.stride = mul i32 %blocks, %index.width
%representative.total = mul i32 %representative.stride, %rows
%score.base = add i32 %representative.base, %representative.total
%score.row.stride = mul i32 %length, %score.stride
%attn.rows.wide = zext i32 %rows to i64 %attn.from.wide = zext i32 %from to i64 %attn.heads.wide = zext i32 %heads to i64 %attn.channels.wide = zext i32 %channels to i64 %attn.length.wide = zext i32 %length to i64 %attn.head.width.wide = zext i32 %head.width to i64 %attn.kv.heads.wide = zext i32 %kv.heads to i64 %attn.index.heads.wide = zext i32 %index.heads to i64 %attn.index.width.wide = zext i32 %index.width to i64
%attn.kv.channels.wide = mul i64 %attn.kv.heads.wide, %attn.head.width.wide %attn.kv.plane.wide = mul i64 %attn.kv.channels.wide, %attn.length.wide %attn.value.heads.wide = zext i32 %value.heads to i64 %attn.value.channels.wide = mul i64 %attn.value.heads.wide, %attn.head.width.wide %attn.value.plane.wide = mul i64 %attn.value.channels.wide, %attn.length.wide %attn.kv.planes.wide = add i64 %attn.kv.plane.wide, %attn.value.plane.wide %attn.value.plane.base.wide = add i64 %attn.from.wide, %attn.kv.plane.wide
%attn.index.query.channels.wide = mul i64 %attn.index.heads.wide, %attn.index.width.wide %attn.index.channels.wide = add i64 %attn.index.query.channels.wide, %attn.index.width.wide %attn.index.plane.wide = mul i64 %attn.index.channels.wide, %attn.length.wide %attn.gate.plane.wide = select i1 %gate, i64 %attn.from.wide, i64 0
%attn.index.query.base.wide = add i64 %attn.from.wide, %attn.kv.planes.wide %attn.gate.base.wide = add i64 %attn.index.query.base.wide, 0 %attn.row.stride.wide = add i64 %attn.gate.base.wide, %attn.gate.plane.wide
%attn.blocks.wide = zext i32 %blocks to i64 %attn.score.stride.wide = mul i64 %attn.blocks.wide, 2 %attn.head.jobs.wide = mul i64 %attn.rows.wide, %attn.heads.wide %attn.statistics.rows.wide = mul i64 %attn.head.jobs.wide, %attn.length.wide %attn.representative.base.wide = mul i64 %attn.statistics.rows.wide, 2 %attn.representative.stride.wide = mul i64 %attn.blocks.wide, %attn.index.width.wide %attn.representative.total.wide = mul i64 %attn.representative.stride.wide, %attn.rows.wide %attn.score.base.wide = add i64 %attn.representative.base.wide, %attn.representative.total.wide %attn.score.row.stride.wide = mul i64 %attn.length.wide, %attn.score.stride.wide
%query.values = mul i32 %tile.m, %head.width
%key.values = mul i32 %tile.n, %head.width
%pair.values = mul i32 %tile.m, %tile.n
%query.tiles.rounded = add i32 %length, %tile.m
%query.tiles.numerator = sub i32 %query.tiles.rounded, 1
%query.tiles = udiv i32 %query.tiles.numerator, %tile.m
%key.tiles.rounded = add i32 %length, %tile.n
%key.tiles.numerator = sub i32 %key.tiles.rounded, 1
%key.tiles = udiv i32 %key.tiles.numerator, %tile.n
%dq.jobs = mul i32 %head.jobs, %query.tiles
%dq.delta.base.shared = add i32 0, %query.values
%dq.gradient.base.shared = add i32 %dq.delta.base.shared, %query.values
%dq.key.base.shared = add i32 %dq.gradient.base.shared, %query.values
%dq.value.base.shared = add i32 %dq.key.base.shared, %key.values
%dq.probability.base.shared = add i32 %dq.value.base.shared, %key.values
%dq.derivative.base.shared = add i32 %dq.probability.base.shared, %pair.values
%dq.product.base.shared = add i32 %dq.derivative.base.shared, %pair.values
br label %dq.job.loop
dq.job.loop:
%dq.job = phi i32 [ %group, %entry ], [ %dq.job.next, %dq.job.finish ]
%dq.job.more = icmp ult i32 %dq.job, %dq.jobs
br i1 %dq.job.more, label %dq.job.prepare, label %dq.exit
dq.job.prepare:
%dq.query.tile = urem i32 %dq.job, %query.tiles
%dq.head.job = udiv i32 %dq.job, %query.tiles
%dq.head = urem i32 %dq.head.job, %heads
%dq.row = udiv i32 %dq.head.job, %heads %dq.head.job.wide = zext i32 %dq.head.job to i64 %dq.row.wide = zext i32 %dq.row to i64
%dq.query.base = mul i32 %dq.query.tile, %tile.m %dq.query.base.wide = zext i32 %dq.query.base to i64
%dq.query.remaining = sub i32 %length, %dq.query.base
%dq.query.short = icmp ult i32 %dq.query.remaining, %tile.m
%dq.query.count = select i1 %dq.query.short, i32 %dq.query.remaining, i32 %tile.m
%dq.query.last = add i32 %dq.query.base, %dq.query.count
%dq.head.start = mul i32 %dq.head, %head.width %dq.head.wide = zext i32 %dq.head to i64 %dq.head.start.wide = mul i64 %dq.head.wide, %attn.head.width.wide
%dq.kv.head = udiv i32 %dq.head, %kv.group %dq.kv.head.wide = zext i32 %dq.kv.head to i64 %dq.kv.head.start = mul i32 %dq.kv.head, %head.width %dq.kv.head.start.wide = mul i64 %dq.kv.head.wide, %attn.head.width.wide
%dq.value.head = udiv i32 %dq.head, %value.group %dq.value.head.start = mul i32 %dq.value.head, %head.width
%dq.input.row = mul i32 %dq.row, %row.stride %dq.input.row.wide = mul i64 %dq.row.wide, %attn.row.stride.wide
%dq.output.row = mul i32 %dq.row, %from %dq.output.row.wide = mul i64 %dq.row.wide, %attn.from.wide
%dq.score.row = mul i32 %dq.row, %score.row.stride %dq.score.row.wide = mul i64 %dq.row.wide, %attn.score.row.stride.wide
%dq.score.row.base = add i64 %attn.score.base.wide, %dq.score.row.wide
%dq.active.query.values = mul i32 %dq.query.count, %head.width
br label %dq.query.stage.loop
dq.query.stage.loop:
%dq.query.p = phi i32 [ %lid, %dq.job.prepare ], [ %dq.query.p.next, %dq.query.stage.step ]
%dq.query.p.more = icmp ult i32 %dq.query.p, %dq.active.query.values
br i1 %dq.query.p.more, label %dq.query.stage.step, label %dq.query.stage.done
dq.query.stage.step:
%dq.query.local = udiv i32 %dq.query.p, %head.width %dq.query.local.wide = zext i32 %dq.query.local to i64
%dq.channel.local = urem i32 %dq.query.p, %head.width %dq.channel.local.wide = zext i32 %dq.channel.local to i64
%dq.query.position = add i32 %dq.query.base, %dq.query.local %dq.query.position.wide = add i64 %dq.query.base.wide, %dq.query.local.wide
%dq.channel = add i32 %dq.head.start, %dq.channel.local %dq.channel.wide = add i64 %dq.head.start.wide, %dq.channel.local.wide
%dq.channel.base = mul i32 %dq.channel, %length %dq.channel.base.wide = mul i64 %dq.channel.wide, %attn.length.wide
%dq.local = add i32 %dq.channel.base, %dq.query.position %dq.local.wide = add i64 %dq.channel.base.wide, %dq.query.position.wide
%dq.input.index = add i32 %dq.input.row, %dq.local %dq.input.index.wide = add i64 %dq.input.row.wide, %dq.local.wide
%dq.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %dq.input.index.wide
%dq.query.value.model = load i8, ptr addrspace(1) %dq.input.ptr, align 1
%dq.query.value = call float @recipe.decode(i8 %dq.query.value.model)
%dq.query.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.query.p
store float %dq.query.value, ptr addrspace(3) %dq.query.shared.ptr, align 4
%dq.delta.index = add i32 %dq.output.row, %dq.local %dq.delta.index.wide = add i64 %dq.output.row.wide, %dq.local.wide
%dq.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %dq.delta.index.wide
%dq.delta.value = load float, ptr addrspace(1) %dq.delta.ptr, align 4
%dq.delta.shared.index = add i32 %dq.delta.base.shared, %dq.query.p
%dq.delta.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.delta.shared.index
store float %dq.delta.value, ptr addrspace(3) %dq.delta.shared.ptr, align 4
%dq.gradient.shared.index = add i32 %dq.gradient.base.shared, %dq.query.p
%dq.gradient.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.gradient.shared.index
store float %state.zero, ptr addrspace(3) %dq.gradient.shared.ptr, align 4
%dq.query.p.next = add i32 %dq.query.p, %block
br label %dq.query.stage.loop
dq.query.stage.done:
call void @llvm.nvvm.barrier0()
call void @attention_tile_products(ptr addrspace(1) %output, i64 %dq.output.row.wide, i32 %dq.delta.base.shared,
i32 %dq.product.base.shared, i32 %dq.query.base, i32 %dq.query.count, i32 %dq.head.start,
i32 %head.width, i32 %length, i32 %lid, i32 %block)
call void @llvm.nvvm.barrier0()
br label %dq.norm.done
dq.norm.done:
br i1 %gate, label %dq.gate.loop, label %dq.gate.done
dq.gate.loop:
%dq.gate.p = phi i32 [ %lid, %dq.norm.done ], [ %dq.gate.p.next, %dq.gate.step ]
%dq.gate.more = icmp ult i32 %dq.gate.p, %dq.active.query.values
br i1 %dq.gate.more, label %dq.gate.step, label %dq.gate.exit
dq.gate.step:
%dq.gate.query = udiv i32 %dq.gate.p, %head.width %dq.gate.query.wide = zext i32 %dq.gate.query to i64
%dq.gate.channel = urem i32 %dq.gate.p, %head.width %dq.gate.channel.wide = zext i32 %dq.gate.channel to i64
%dq.gate.position = add i32 %dq.query.base, %dq.gate.query %dq.gate.position.wide = add i64 %dq.query.base.wide, %dq.gate.query.wide
%dq.gate.output.channel = add i32 %dq.head.start, %dq.gate.channel %dq.gate.output.channel.wide = add i64 %dq.head.start.wide, %dq.gate.channel.wide
%dq.gate.channel.base = mul i32 %dq.gate.output.channel, %length %dq.gate.channel.base.wide = mul i64 %dq.gate.output.channel.wide, %attn.length.wide
%dq.gate.local = add i32 %dq.gate.channel.base, %dq.gate.position %dq.gate.local.wide = add i64 %dq.gate.channel.base.wide, %dq.gate.position.wide
%dq.gate.row = add i32 %dq.input.row, %gate.base %dq.gate.row.wide = add i64 %dq.input.row.wide, %attn.gate.base.wide
%dq.gate.index = add i32 %dq.gate.row, %dq.gate.local %dq.gate.index.wide = add i64 %dq.gate.row.wide, %dq.gate.local.wide
%dq.gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %dq.gate.index.wide
%dq.gate.value = load i8, ptr addrspace(1) %dq.gate.ptr, align 1
%dq.gate.factor.model = call i8 @recipe.sigmoid(i8 %dq.gate.value)
%dq.gate.factor = call float @recipe.decode(i8 %dq.gate.factor.model)
%dq.gate.shared.index = add i32 %dq.delta.base.shared, %dq.gate.p
%dq.gate.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.gate.shared.index
%dq.gate.delta = load float, ptr addrspace(3) %dq.gate.shared.ptr, align 4
%dq.gate.output.index = add i32 %dq.output.row, %dq.gate.local %dq.gate.output.index.wide = add i64 %dq.output.row.wide, %dq.gate.local.wide
%dq.gate.output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %dq.gate.output.index.wide
%dq.gate.output.value.model = load i8, ptr addrspace(1) %dq.gate.output.ptr, align 1
%dq.gate.output.value = call float @recipe.decode(i8 %dq.gate.output.value.model)
%dq.gate.one = call float @recipe.state.from.u1(i1 true)
%dq.gate.complement = call float @recipe.state.sub(float %dq.gate.one, float %dq.gate.factor)
%dq.gate.product = call float @recipe.state.mul(float %dq.gate.delta, float %dq.gate.output.value)
%dq.gate.gradient = call float @recipe.state.mul(float %dq.gate.product, float %dq.gate.complement)
%dq.gate.previous.ptr = getelementptr inbounds float, ptr addrspace(1) %previous, i64 %dq.gate.index.wide
store float %dq.gate.gradient, ptr addrspace(1) %dq.gate.previous.ptr, align 4
%dq.gate.scaled = call float @recipe.state.mul(float %dq.gate.delta, float %dq.gate.factor)
store float %dq.gate.scaled, ptr addrspace(3) %dq.gate.shared.ptr, align 4
%dq.gate.p.next = add i32 %dq.gate.p, %block
br label %dq.gate.loop
dq.gate.exit:
call void @llvm.nvvm.barrier0()
br label %dq.gate.done
dq.gate.done:
br label %dq.key.tile.loop
dq.key.tile.loop:
%dq.key.tile.base = phi i32 [ 0, %dq.gate.done ], [ %dq.key.tile.next, %dq.key.tile.advance ]
%dq.key.tile.more = icmp ult i32 %dq.key.tile.base, %dq.query.last
br i1 %dq.key.tile.more, label %dq.key.tile.prepare, label %dq.store.begin
dq.key.tile.prepare:
%dq.key.remaining = sub i32 %dq.query.last, %dq.key.tile.base
%dq.key.short = icmp ult i32 %dq.key.remaining, %tile.n
%dq.key.count = select i1 %dq.key.short, i32 %dq.key.remaining, i32 %tile.n
%dq.active.key.values = mul i32 %dq.key.count, %head.width
br i1 %select, label %dq.scan.prepare, label %dq.key.stage.loop
dq.scan.prepare:
%dq.scan.first.block = udiv i32 %dq.key.tile.base, %select.block
%dq.scan.stop = add i32 %dq.key.tile.base, %dq.key.count
%dq.scan.stop.less = sub i32 %dq.scan.stop, 1
%dq.scan.last.block = udiv i32 %dq.scan.stop.less, %select.block
br label %dq.scan.loop
dq.scan.loop:
%dq.scan.q = phi i32 [ 0, %dq.scan.prepare ], [ %dq.scan.q.next, %dq.scan.block.done ]
%dq.scan.more = icmp ult i32 %dq.scan.q, %dq.query.count
br i1 %dq.scan.more, label %dq.scan.query, label %dq.key.tile.advance
dq.scan.query:
%dq.scan.query.index = add i32 %dq.query.base, %dq.scan.q
br label %dq.scan.block.loop
dq.scan.block.loop:
%dq.scan.b = phi i32 [ %dq.scan.first.block, %dq.scan.query ], [ %dq.scan.b.next, %dq.scan.block.advance ]
%dq.scan.block.more = icmp ule i32 %dq.scan.b, %dq.scan.last.block
br i1 %dq.scan.block.more, label %dq.scan.block.step, label %dq.scan.block.done
dq.scan.block.step:
%dq.scan.block.start = mul i32 %dq.scan.b, %select.block
%dq.scan.before = icmp ult i32 %dq.scan.block.start, %dq.key.tile.base
%dq.scan.key = select i1 %dq.scan.before, i32 %dq.key.tile.base, i32 %dq.scan.block.start
%dq.scan.causal = icmp ule i32 %dq.scan.key, %dq.scan.query.index
%dq.scan.kept = call i1 @attention_selected(ptr addrspace(1) %context, i64 %dq.score.row.base, i32 %blocks, i32 %select.block, i32 %dq.scan.query.index, i32 %dq.scan.key)
%dq.scan.hit = and i1 %dq.scan.causal, %dq.scan.kept
br i1 %dq.scan.hit, label %dq.key.stage.loop, label %dq.scan.block.advance
dq.scan.block.advance:
%dq.scan.b.next = add i32 %dq.scan.b, 1
br label %dq.scan.block.loop
dq.scan.block.done:
%dq.scan.q.next = add i32 %dq.scan.q, 1
br label %dq.scan.loop
dq.key.stage.loop:
%dq.key.p = phi i32 [ %lid, %dq.key.tile.prepare ], [ %lid, %dq.scan.block.step ], [ %dq.key.p.next, %dq.key.stage.step ]
%dq.key.p.more = icmp ult i32 %dq.key.p, %dq.active.key.values
br i1 %dq.key.p.more, label %dq.key.stage.step, label %dq.key.stage.done
dq.key.stage.step:
%dq.key.local = udiv i32 %dq.key.p, %head.width
%dq.key.channel.local = urem i32 %dq.key.p, %head.width
%dq.key.position = add i32 %dq.key.tile.base, %dq.key.local
%dq.key.channel = add i32 %dq.kv.head.start, %dq.key.channel.local
%dq.key.channel.base = mul i32 %dq.key.channel, %length
%dq.key.input.local = add i32 %dq.key.channel.base, %dq.key.position
%dq.key.plane = add i32 %dq.input.row, %from
%dq.key.input.index = add i32 %dq.key.plane, %dq.key.input.local
%dq.key.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i32 %dq.key.input.index
%dq.key.value.model = load i8, ptr addrspace(1) %dq.key.input.ptr, align 1
%dq.key.value = call float @recipe.decode(i8 %dq.key.value.model)
%dq.key.shared.index = add i32 %dq.key.base.shared, %dq.key.p
%dq.key.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.key.shared.index
store float %dq.key.value, ptr addrspace(3) %dq.key.shared.ptr, align 4
%dq.value.row = add i32 %dq.input.row, %value.plane.base
%dq.value.channel = add i32 %dq.value.head.start, %dq.key.channel.local
%dq.value.channel.base = mul i32 %dq.value.channel, %length
%dq.value.input.local = add i32 %dq.value.channel.base, %dq.key.position
%dq.value.input.index = add i32 %dq.value.row, %dq.value.input.local
%dq.value.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i32 %dq.value.input.index
%dq.value.value.model = load i8, ptr addrspace(1) %dq.value.input.ptr, align 1
%dq.value.value = call float @recipe.decode(i8 %dq.value.value.model)
%dq.value.shared.index = add i32 %dq.value.base.shared, %dq.key.p
%dq.value.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.value.shared.index
store float %dq.value.value, ptr addrspace(3) %dq.value.shared.ptr, align 4
%dq.key.p.next = add i32 %dq.key.p, %block
br label %dq.key.stage.loop
dq.key.stage.done:
call void @llvm.nvvm.barrier0()
br label %dq.key.norm.done
dq.key.norm.done:
%dq.head.job.arg = zext i32 %dq.head.job to i64
%dq.statistics.denominator.base.arg = zext i32 %statistics.denominator.base to i64
call void @attention_tile_derivatives(ptr addrspace(1) %context, i32 0, i32 %dq.key.base.shared,
i32 %dq.delta.base.shared, i32 %dq.value.base.shared, i32 %dq.probability.base.shared,
i32 %dq.derivative.base.shared, i32 %dq.product.base.shared, i32 %dq.query.base,
i32 %dq.key.tile.base, i32 %dq.query.count, i32 %dq.key.count, i32 %tile.n,
i64 %dq.head.job.arg, i64 %attn.length.wide, i64 %dq.statistics.denominator.base.arg, i32 %head.width,
i8 %scale, i32 %lid, i32 %block, i64 %dq.score.row.base, i32 %blocks, i32 %select.block, i1 %select)
call void @llvm.nvvm.barrier0()
br label %dq.accumulate.loop
dq.accumulate.loop:
%dq.accumulate.p = phi i32 [ %lid, %dq.key.norm.done ], [ %dq.accumulate.p.next, %dq.accumulate.store ]
%dq.accumulate.p.more = icmp ult i32 %dq.accumulate.p, %dq.active.query.values
br i1 %dq.accumulate.p.more, label %dq.accumulate.prepare, label %dq.accumulate.done
dq.accumulate.prepare:
%dq.accumulate.query = udiv i32 %dq.accumulate.p, %head.width
%dq.accumulate.channel = urem i32 %dq.accumulate.p, %head.width
%dq.accumulate.gradient.index = add i32 %dq.gradient.base.shared, %dq.accumulate.p
%dq.accumulate.gradient.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.accumulate.gradient.index
%dq.accumulate.initial = load float, ptr addrspace(3) %dq.accumulate.gradient.ptr, align 4
br label %dq.accumulate.key.loop
dq.accumulate.key.loop:
%dq.accumulate.key = phi i32 [ 0, %dq.accumulate.prepare ], [ %dq.accumulate.key.next, %dq.accumulate.key.step ]
%dq.accumulate.value = phi float [ %dq.accumulate.initial, %dq.accumulate.prepare ], [ %dq.accumulate.next, %dq.accumulate.key.step ]
%dq.accumulate.key.more = icmp ult i32 %dq.accumulate.key, %dq.key.count
br i1 %dq.accumulate.key.more, label %dq.accumulate.key.step, label %dq.accumulate.store
dq.accumulate.key.step:
%dq.accumulate.pair.row = mul i32 %dq.accumulate.query, %tile.n
%dq.accumulate.pair.local = add i32 %dq.accumulate.pair.row, %dq.accumulate.key
%dq.accumulate.pair.index = add i32 %dq.derivative.base.shared, %dq.accumulate.pair.local
%dq.accumulate.pair.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.accumulate.pair.index
%dq.accumulate.ds = load float, ptr addrspace(3) %dq.accumulate.pair.ptr, align 4
%dq.accumulate.key.row = mul i32 %dq.accumulate.key, %head.width
%dq.accumulate.key.local = add i32 %dq.accumulate.key.row, %dq.accumulate.channel
%dq.accumulate.key.index = add i32 %dq.key.base.shared, %dq.accumulate.key.local
%dq.accumulate.key.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.accumulate.key.index
%dq.accumulate.key.value = load float, ptr addrspace(3) %dq.accumulate.key.ptr, align 4
%dq.accumulate.raw = call float @recipe.state.mul(float %dq.accumulate.ds, float %dq.accumulate.key.value)
%dq.accumulate.term = call float @recipe.state.div(float %dq.accumulate.raw, float %scale.wide)
%dq.accumulate.next = call float @recipe.state.add(float %dq.accumulate.value, float %dq.accumulate.term)
%dq.accumulate.key.next = add i32 %dq.accumulate.key, 1
br label %dq.accumulate.key.loop
dq.accumulate.store:
store float %dq.accumulate.value, ptr addrspace(3) %dq.accumulate.gradient.ptr, align 4
%dq.accumulate.p.next = add i32 %dq.accumulate.p, %block
br label %dq.accumulate.loop
dq.accumulate.done:
call void @llvm.nvvm.barrier0()
br label %dq.key.tile.advance
dq.key.tile.advance:
%dq.key.tile.next = add i32 %dq.key.tile.base, %tile.n
br label %dq.key.tile.loop
dq.store.begin:
call void @llvm.nvvm.barrier0()
br label %dq.adjoint.done
dq.adjoint.done:
br label %dq.store.loop
dq.store.loop:
%dq.store.p = phi i32 [ %lid, %dq.adjoint.done ], [ %dq.store.p.next, %dq.store.step ]
%dq.store.p.more = icmp ult i32 %dq.store.p, %dq.active.query.values
br i1 %dq.store.p.more, label %dq.store.step, label %dq.store.done
dq.store.step:
%dq.store.query.local = udiv i32 %dq.store.p, %head.width
%dq.store.channel.local = urem i32 %dq.store.p, %head.width
%dq.store.query = add i32 %dq.query.base, %dq.store.query.local
%dq.store.channel = add i32 %dq.head.start, %dq.store.channel.local
%dq.store.channel.base = mul i32 %dq.store.channel, %length
%dq.store.local = add i32 %dq.store.channel.base, %dq.store.query
%dq.store.index = add i32 %dq.input.row, %dq.store.local
%dq.store.ptr = getelementptr inbounds float, ptr addrspace(1) %previous, i32 %dq.store.index
%dq.store.shared.index = add i32 %dq.gradient.base.shared, %dq.store.p
%dq.store.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dq.store.shared.index
%dq.store.value = load float, ptr addrspace(3) %dq.store.shared.ptr, align 4
store float %dq.store.value, ptr addrspace(1) %dq.store.ptr, align 4
%dq.store.p.next = add i32 %dq.store.p, %block
br label %dq.store.loop
dq.store.done:
call void @llvm.nvvm.barrier0()
br label %dq.job.finish
dq.job.finish:
%dq.job.next = add i32 %dq.job, %groups
br label %dq.job.loop
dq.exit:
%dkv.total.heads = add i32 %kv.heads, %value.heads
%dkv.head.jobs = mul i32 %rows, %dkv.total.heads
%dkv.jobs = mul i32 %dkv.head.jobs, %key.tiles
%dkv.value.base.shared = add i32 0, %key.values
%dkv.key.gradient.base.shared = add i32 %dkv.value.base.shared, %key.values
%dkv.value.gradient.base.shared = add i32 %dkv.key.gradient.base.shared, %key.values
%dkv.query.base.shared = add i32 %dkv.value.gradient.base.shared, %key.values
%dkv.delta.base.shared = add i32 %dkv.query.base.shared, %query.values
%dkv.probability.base.shared = add i32 %dkv.delta.base.shared, %query.values
%dkv.derivative.base.shared = add i32 %dkv.probability.base.shared, %pair.values
%dkv.product.base.shared = add i32 %dkv.derivative.base.shared, %pair.values
br label %dkv.job.loop
dkv.job.loop:
%dkv.job = phi i32 [ %group, %dq.exit ], [ %dkv.job.next, %dkv.job.finish ]
%dkv.job.more = icmp ult i32 %dkv.job, %dkv.jobs
br i1 %dkv.job.more, label %dkv.job.prepare, label %exit
dkv.job.prepare:
%dkv.key.tile = urem i32 %dkv.job, %key.tiles
%dkv.kv.job = udiv i32 %dkv.job, %key.tiles
; Key jobs own one key head and its query group; value jobs own one value head and its query group.
%dkv.fixed.head = urem i32 %dkv.kv.job, %dkv.total.heads
%dkv.row = udiv i32 %dkv.kv.job, %dkv.total.heads
%dkv.is.key = icmp ult i32 %dkv.fixed.head, %kv.heads
%dkv.fixed.value.head.raw = sub i32 %dkv.fixed.head, %kv.heads
%dkv.fixed.value.head = select i1 %dkv.is.key, i32 0, i32 %dkv.fixed.value.head.raw
%dkv.kv.head = select i1 %dkv.is.key, i32 %dkv.fixed.head, i32 0
%dkv.key.base = mul i32 %dkv.key.tile, %tile.n
%dkv.key.remaining = sub i32 %length, %dkv.key.base
%dkv.key.short = icmp ult i32 %dkv.key.remaining, %tile.n
%dkv.key.count = select i1 %dkv.key.short, i32 %dkv.key.remaining, i32 %tile.n
%dkv.kv.head.start = mul i32 %dkv.kv.head, %head.width
%dkv.store.value.head.start = mul i32 %dkv.fixed.value.head, %head.width
%dkv.input.row = mul i32 %dkv.row, %row.stride
%dkv.output.row = mul i32 %dkv.row, %from
%dkv.score.row = mul i32 %dkv.row, %score.row.stride
%dkv.score.row.base.narrow = add i32 %score.base, %dkv.score.row
%dkv.score.row.base = zext i32 %dkv.score.row.base.narrow to i64
%dkv.active.key.values = mul i32 %dkv.key.count, %head.width
%dkv.head.row = mul i32 %dkv.row, %heads
%dkv.head.base.key = mul i32 %dkv.fixed.head, %kv.group
%dkv.head.base.value = mul i32 %dkv.fixed.value.head, %value.group
%dkv.head.base = select i1 %dkv.is.key, i32 %dkv.head.base.key, i32 %dkv.head.base.value
%dkv.head.count = select i1 %dkv.is.key, i32 %kv.group, i32 %value.group
br label %dkv.zero.loop
dkv.zero.loop:
%dkv.zero.p = phi i32 [ %lid, %dkv.job.prepare ], [ %dkv.zero.p.next, %dkv.zero.step ]
%dkv.zero.p.more = icmp ult i32 %dkv.zero.p, %dkv.active.key.values
br i1 %dkv.zero.p.more, label %dkv.zero.step, label %dkv.zero.done
dkv.zero.step:
%dkv.key.gradient.index = add i32 %dkv.key.gradient.base.shared, %dkv.zero.p
%dkv.key.gradient.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.key.gradient.index
store float %state.zero, ptr addrspace(3) %dkv.key.gradient.ptr, align 4
%dkv.value.gradient.index = add i32 %dkv.value.gradient.base.shared, %dkv.zero.p
%dkv.value.gradient.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.value.gradient.index
store float %state.zero, ptr addrspace(3) %dkv.value.gradient.ptr, align 4
%dkv.zero.p.next = add i32 %dkv.zero.p, %block
br label %dkv.zero.loop
dkv.zero.done:
call void @llvm.nvvm.barrier0()
br label %dkv.head.loop
dkv.head.loop:
%dkv.head.slot = phi i32 [ 0, %dkv.zero.done ], [ %dkv.head.slot.next, %dkv.head.step ]
%dkv.head.more = icmp ult i32 %dkv.head.slot, %dkv.head.count
br i1 %dkv.head.more, label %dkv.head.prepare, label %dkv.store.begin
dkv.head.prepare:
%dkv.head = add i32 %dkv.head.base, %dkv.head.slot
%dkv.head.start = mul i32 %dkv.head, %head.width
%dkv.head.job = add i32 %dkv.head.row, %dkv.head
; The tiles restage per query head: a key job reads that head's value head, a value job that head's key head.
%dkv.head.key.head.query = udiv i32 %dkv.head, %kv.group
%dkv.head.value.head.query = udiv i32 %dkv.head, %value.group
%dkv.head.key.head = select i1 %dkv.is.key, i32 %dkv.fixed.head, i32 %dkv.head.key.head.query
%dkv.head.value.head = select i1 %dkv.is.key, i32 %dkv.head.value.head.query, i32 %dkv.fixed.value.head
%dkv.head.key.start = mul i32 %dkv.head.key.head, %head.width
%dkv.head.value.start = mul i32 %dkv.head.value.head, %head.width
call void @llvm.nvvm.barrier0()
br label %dkv.key.stage.loop
dkv.key.stage.loop:
%dkv.key.p = phi i32 [ %lid, %dkv.head.prepare ], [ %dkv.key.p.next, %dkv.key.stage.step ]
%dkv.key.p.more = icmp ult i32 %dkv.key.p, %dkv.active.key.values
br i1 %dkv.key.p.more, label %dkv.key.stage.step, label %dkv.key.stage.done
dkv.key.stage.step:
%dkv.key.local = udiv i32 %dkv.key.p, %head.width
%dkv.channel.local = urem i32 %dkv.key.p, %head.width
%dkv.key.position = add i32 %dkv.key.base, %dkv.key.local
%dkv.channel = add i32 %dkv.head.key.start, %dkv.channel.local
%dkv.channel.base = mul i32 %dkv.channel, %length
%dkv.local = add i32 %dkv.channel.base, %dkv.key.position
%dkv.key.plane = add i32 %dkv.input.row, %from
%dkv.key.input.index = add i32 %dkv.key.plane, %dkv.local
%dkv.key.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i32 %dkv.key.input.index
%dkv.key.value.model = load i8, ptr addrspace(1) %dkv.key.input.ptr, align 1
%dkv.key.value = call float @recipe.decode(i8 %dkv.key.value.model)
%dkv.key.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.key.p
store float %dkv.key.value, ptr addrspace(3) %dkv.key.shared.ptr, align 4
%dkv.value.channel = add i32 %dkv.head.value.start, %dkv.channel.local
%dkv.value.channel.base = mul i32 %dkv.value.channel, %length
%dkv.value.local = add i32 %dkv.value.channel.base, %dkv.key.position
%dkv.value.row = add i32 %dkv.input.row, %value.plane.base
%dkv.value.input.index = add i32 %dkv.value.row, %dkv.value.local
%dkv.value.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i32 %dkv.value.input.index
%dkv.value.value.model = load i8, ptr addrspace(1) %dkv.value.input.ptr, align 1
%dkv.value.value = call float @recipe.decode(i8 %dkv.value.value.model)
%dkv.value.shared.index = add i32 %dkv.value.base.shared, %dkv.key.p
%dkv.value.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.value.shared.index
store float %dkv.value.value, ptr addrspace(3) %dkv.value.shared.ptr, align 4
%dkv.key.p.next = add i32 %dkv.key.p, %block
br label %dkv.key.stage.loop
dkv.key.stage.done:
call void @llvm.nvvm.barrier0()
br label %dkv.key.norm.done
dkv.key.norm.done:
br label %dkv.query.tile.loop
dkv.query.tile.loop:
%dkv.query.base = phi i32 [ %dkv.key.base, %dkv.key.norm.done ], [ %dkv.query.next, %dkv.query.advance ]
%dkv.query.more = icmp ult i32 %dkv.query.base, %length
br i1 %dkv.query.more, label %dkv.query.tile.prepare, label %dkv.head.step
dkv.query.tile.prepare:
%dkv.query.remaining = sub i32 %length, %dkv.query.base
%dkv.query.short = icmp ult i32 %dkv.query.remaining, %tile.m
%dkv.query.count = select i1 %dkv.query.short, i32 %dkv.query.remaining, i32 %tile.m
%dkv.active.query.values = mul i32 %dkv.query.count, %head.width
br i1 %select, label %dkv.scan.prepare, label %dkv.query.stage.loop
dkv.scan.prepare:
%dkv.scan.first.block = udiv i32 %dkv.key.base, %select.block
%dkv.scan.stop = add i32 %dkv.key.base, %dkv.key.count
%dkv.scan.stop.less = sub i32 %dkv.scan.stop, 1
%dkv.scan.last.block = udiv i32 %dkv.scan.stop.less, %select.block
br label %dkv.scan.loop
dkv.scan.loop:
%dkv.scan.q = phi i32 [ 0, %dkv.scan.prepare ], [ %dkv.scan.q.next, %dkv.scan.block.done ]
%dkv.scan.more = icmp ult i32 %dkv.scan.q, %dkv.query.count
br i1 %dkv.scan.more, label %dkv.scan.query, label %dkv.query.advance
dkv.scan.query:
%dkv.scan.query.index = add i32 %dkv.query.base, %dkv.scan.q
br label %dkv.scan.block.loop
dkv.scan.block.loop:
%dkv.scan.b = phi i32 [ %dkv.scan.first.block, %dkv.scan.query ], [ %dkv.scan.b.next, %dkv.scan.block.advance ]
%dkv.scan.block.more = icmp ule i32 %dkv.scan.b, %dkv.scan.last.block
br i1 %dkv.scan.block.more, label %dkv.scan.block.step, label %dkv.scan.block.done
dkv.scan.block.step:
%dkv.scan.block.start = mul i32 %dkv.scan.b, %select.block
%dkv.scan.before = icmp ult i32 %dkv.scan.block.start, %dkv.key.base
%dkv.scan.key = select i1 %dkv.scan.before, i32 %dkv.key.base, i32 %dkv.scan.block.start
%dkv.scan.causal = icmp ule i32 %dkv.scan.key, %dkv.scan.query.index
%dkv.scan.kept = call i1 @attention_selected(ptr addrspace(1) %context, i64 %dkv.score.row.base, i32 %blocks, i32 %select.block, i32 %dkv.scan.query.index, i32 %dkv.scan.key)
%dkv.scan.hit = and i1 %dkv.scan.causal, %dkv.scan.kept
br i1 %dkv.scan.hit, label %dkv.query.stage.loop, label %dkv.scan.block.advance
dkv.scan.block.advance:
%dkv.scan.b.next = add i32 %dkv.scan.b, 1
br label %dkv.scan.block.loop
dkv.scan.block.done:
%dkv.scan.q.next = add i32 %dkv.scan.q, 1
br label %dkv.scan.loop
dkv.query.stage.loop:
%dkv.query.p = phi i32 [ %lid, %dkv.query.tile.prepare ], [ %lid, %dkv.scan.block.step ], [ %dkv.query.p.next, %dkv.query.stage.step ]
%dkv.query.p.more = icmp ult i32 %dkv.query.p, %dkv.active.query.values
br i1 %dkv.query.p.more, label %dkv.query.stage.step, label %dkv.query.stage.done
dkv.query.stage.step:
%dkv.query.local = udiv i32 %dkv.query.p, %head.width
%dkv.query.channel.local = urem i32 %dkv.query.p, %head.width
%dkv.query.position = add i32 %dkv.query.base, %dkv.query.local
%dkv.query.channel = add i32 %dkv.head.start, %dkv.query.channel.local
%dkv.query.channel.base = mul i32 %dkv.query.channel, %length
%dkv.query.input.local = add i32 %dkv.query.channel.base, %dkv.query.position
%dkv.query.input.index = add i32 %dkv.input.row, %dkv.query.input.local
%dkv.query.input.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i32 %dkv.query.input.index
%dkv.query.value.model = load i8, ptr addrspace(1) %dkv.query.input.ptr, align 1
%dkv.query.value = call float @recipe.decode(i8 %dkv.query.value.model)
%dkv.query.shared.index = add i32 %dkv.query.base.shared, %dkv.query.p
%dkv.query.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.query.shared.index
store float %dkv.query.value, ptr addrspace(3) %dkv.query.shared.ptr, align 4
%dkv.delta.input.index = add i32 %dkv.output.row, %dkv.query.input.local
%dkv.delta.input.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i32 %dkv.delta.input.index
%dkv.delta.value = load float, ptr addrspace(1) %dkv.delta.input.ptr, align 4
%dkv.delta.shared.index = add i32 %dkv.delta.base.shared, %dkv.query.p
%dkv.delta.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.delta.shared.index
store float %dkv.delta.value, ptr addrspace(3) %dkv.delta.shared.ptr, align 4
%dkv.query.p.next = add i32 %dkv.query.p, %block
br label %dkv.query.stage.loop
dkv.query.stage.done:
call void @llvm.nvvm.barrier0()
%dkv.output.row.wide.arg = zext i32 %dkv.output.row to i64
call void @attention_tile_products(ptr addrspace(1) %output, i64 %dkv.output.row.wide.arg, i32 %dkv.delta.base.shared,
i32 %dkv.product.base.shared, i32 %dkv.query.base, i32 %dkv.query.count, i32 %dkv.head.start,
i32 %head.width, i32 %length, i32 %lid, i32 %block)
call void @llvm.nvvm.barrier0()
br label %dkv.query.norm.done
dkv.query.norm.done:
br i1 %gate, label %dkv.gate.loop, label %dkv.gate.done
dkv.gate.loop:
%dkv.gate.p = phi i32 [ %lid, %dkv.query.norm.done ], [ %dkv.gate.p.next, %dkv.gate.step ]
%dkv.gate.more = icmp ult i32 %dkv.gate.p, %dkv.active.query.values
br i1 %dkv.gate.more, label %dkv.gate.step, label %dkv.gate.exit
dkv.gate.step:
%dkv.gate.query = udiv i32 %dkv.gate.p, %head.width
%dkv.gate.channel = urem i32 %dkv.gate.p, %head.width
%dkv.gate.position = add i32 %dkv.query.base, %dkv.gate.query
%dkv.gate.output.channel = add i32 %dkv.head.start, %dkv.gate.channel
%dkv.gate.channel.base = mul i32 %dkv.gate.output.channel, %length
%dkv.gate.local = add i32 %dkv.gate.channel.base, %dkv.gate.position
%dkv.gate.row = add i32 %dkv.input.row, %gate.base
%dkv.gate.index = add i32 %dkv.gate.row, %dkv.gate.local
%dkv.gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %input, i32 %dkv.gate.index
%dkv.gate.value = load i8, ptr addrspace(1) %dkv.gate.ptr, align 1
%dkv.gate.factor.model = call i8 @recipe.sigmoid(i8 %dkv.gate.value)
%dkv.gate.factor = call float @recipe.decode(i8 %dkv.gate.factor.model)
%dkv.gate.shared.index = add i32 %dkv.delta.base.shared, %dkv.gate.p
%dkv.gate.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.gate.shared.index
%dkv.gate.delta = load float, ptr addrspace(3) %dkv.gate.shared.ptr, align 4
%dkv.gate.scaled = call float @recipe.state.mul(float %dkv.gate.delta, float %dkv.gate.factor)
store float %dkv.gate.scaled, ptr addrspace(3) %dkv.gate.shared.ptr, align 4
%dkv.gate.p.next = add i32 %dkv.gate.p, %block
br label %dkv.gate.loop
dkv.gate.exit:
call void @llvm.nvvm.barrier0()
br label %dkv.gate.done
dkv.gate.done:
%dkv.head.job.arg = zext i32 %dkv.head.job to i64
%dkv.statistics.denominator.base.arg = zext i32 %statistics.denominator.base to i64
call void @attention_tile_derivatives(ptr addrspace(1) %context, i32 %dkv.query.base.shared, i32 0,
i32 %dkv.delta.base.shared, i32 %dkv.value.base.shared, i32 %dkv.probability.base.shared,
i32 %dkv.derivative.base.shared, i32 %dkv.product.base.shared, i32 %dkv.query.base,
i32 %dkv.key.base, i32 %dkv.query.count, i32 %dkv.key.count, i32 %tile.n,
i64 %dkv.head.job.arg, i64 %attn.length.wide, i64 %dkv.statistics.denominator.base.arg, i32 %head.width,
i8 %scale, i32 %lid, i32 %block, i64 %dkv.score.row.base, i32 %blocks, i32 %select.block, i1 %select)
call void @llvm.nvvm.barrier0()
br label %dkv.accumulate.loop
dkv.accumulate.loop:
%dkv.accumulate.p = phi i32 [ %lid, %dkv.gate.done ], [ %dkv.accumulate.p.next, %dkv.accumulate.store ]
%dkv.accumulate.p.more = icmp ult i32 %dkv.accumulate.p, %dkv.active.key.values
br i1 %dkv.accumulate.p.more, label %dkv.accumulate.prepare, label %dkv.accumulate.done
dkv.accumulate.prepare:
%dkv.accumulate.key = udiv i32 %dkv.accumulate.p, %head.width
%dkv.accumulate.channel = urem i32 %dkv.accumulate.p, %head.width
%dkv.accumulate.key.gradient.index = add i32 %dkv.key.gradient.base.shared, %dkv.accumulate.p
%dkv.accumulate.key.gradient.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.accumulate.key.gradient.index
%dkv.accumulate.key.initial = load float, ptr addrspace(3) %dkv.accumulate.key.gradient.ptr, align 4
%dkv.accumulate.value.gradient.index = add i32 %dkv.value.gradient.base.shared, %dkv.accumulate.p
%dkv.accumulate.value.gradient.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.accumulate.value.gradient.index
%dkv.accumulate.value.initial = load float, ptr addrspace(3) %dkv.accumulate.value.gradient.ptr, align 4
br label %dkv.accumulate.query.loop
dkv.accumulate.query.loop:
%dkv.accumulate.query = phi i32 [ 0, %dkv.accumulate.prepare ], [ %dkv.accumulate.query.next, %dkv.accumulate.query.step ]
%dkv.accumulate.key.value = phi float [ %dkv.accumulate.key.initial, %dkv.accumulate.prepare ], [ %dkv.accumulate.key.next, %dkv.accumulate.query.step ]
%dkv.accumulate.value.value = phi float [ %dkv.accumulate.value.initial, %dkv.accumulate.prepare ], [ %dkv.accumulate.value.next, %dkv.accumulate.query.step ]
%dkv.accumulate.query.more = icmp ult i32 %dkv.accumulate.query, %dkv.query.count
br i1 %dkv.accumulate.query.more, label %dkv.accumulate.query.step, label %dkv.accumulate.store
dkv.accumulate.query.step:
%dkv.accumulate.pair.row = mul i32 %dkv.accumulate.query, %tile.n
%dkv.accumulate.pair.local = add i32 %dkv.accumulate.pair.row, %dkv.accumulate.key
%dkv.accumulate.probability.index = add i32 %dkv.probability.base.shared, %dkv.accumulate.pair.local
%dkv.accumulate.probability.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.accumulate.probability.index
%dkv.accumulate.probability = load float, ptr addrspace(3) %dkv.accumulate.probability.ptr, align 4
%dkv.accumulate.derivative.index = add i32 %dkv.derivative.base.shared, %dkv.accumulate.pair.local
%dkv.accumulate.derivative.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.accumulate.derivative.index
%dkv.accumulate.derivative = load float, ptr addrspace(3) %dkv.accumulate.derivative.ptr, align 4
%dkv.accumulate.query.row = mul i32 %dkv.accumulate.query, %head.width
%dkv.accumulate.query.local = add i32 %dkv.accumulate.query.row, %dkv.accumulate.channel
%dkv.accumulate.query.index = add i32 %dkv.query.base.shared, %dkv.accumulate.query.local
%dkv.accumulate.query.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.accumulate.query.index
%dkv.accumulate.query.value = load float, ptr addrspace(3) %dkv.accumulate.query.ptr, align 4
%dkv.accumulate.delta.index = add i32 %dkv.delta.base.shared, %dkv.accumulate.query.local
%dkv.accumulate.delta.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.accumulate.delta.index
%dkv.accumulate.delta = load float, ptr addrspace(3) %dkv.accumulate.delta.ptr, align 4
%dkv.accumulate.key.raw = call float @recipe.state.mul(float %dkv.accumulate.derivative, float %dkv.accumulate.query.value)
%dkv.accumulate.key.term = call float @recipe.state.div(float %dkv.accumulate.key.raw, float %scale.wide)
%dkv.accumulate.key.next = call float @recipe.state.add(float %dkv.accumulate.key.value, float %dkv.accumulate.key.term)
%dkv.accumulate.value.term = call float @recipe.state.mul(float %dkv.accumulate.probability, float %dkv.accumulate.delta)
%dkv.accumulate.value.next = call float @recipe.state.add(float %dkv.accumulate.value.value, float %dkv.accumulate.value.term)
%dkv.accumulate.query.next = add i32 %dkv.accumulate.query, 1
br label %dkv.accumulate.query.loop
dkv.accumulate.store:
store float %dkv.accumulate.key.value, ptr addrspace(3) %dkv.accumulate.key.gradient.ptr, align 4
store float %dkv.accumulate.value.value, ptr addrspace(3) %dkv.accumulate.value.gradient.ptr, align 4
%dkv.accumulate.p.next = add i32 %dkv.accumulate.p, %block
br label %dkv.accumulate.loop
dkv.accumulate.done:
call void @llvm.nvvm.barrier0()
br label %dkv.query.advance
dkv.query.advance:
%dkv.query.next = add i32 %dkv.query.base, %tile.m
br label %dkv.query.tile.loop
dkv.head.step:
%dkv.head.slot.next = add i32 %dkv.head.slot, 1
br label %dkv.head.loop
dkv.store.begin:
call void @llvm.nvvm.barrier0()
br label %dkv.adjoint.done
dkv.adjoint.done:
br label %dkv.store.loop
dkv.store.loop:
%dkv.store.p = phi i32 [ %lid, %dkv.adjoint.done ], [ %dkv.store.p.next, %dkv.store.next ]
%dkv.store.p.more = icmp ult i32 %dkv.store.p, %dkv.active.key.values
br i1 %dkv.store.p.more, label %dkv.store.step, label %dkv.store.done
dkv.store.step:
%dkv.store.key.local = udiv i32 %dkv.store.p, %head.width
%dkv.store.channel.local = urem i32 %dkv.store.p, %head.width
%dkv.store.key = add i32 %dkv.key.base, %dkv.store.key.local
br i1 %dkv.is.key, label %dkv.store.key.step, label %dkv.store.value.step
dkv.store.key.step:
%dkv.store.channel = add i32 %dkv.kv.head.start, %dkv.store.channel.local
%dkv.store.channel.base = mul i32 %dkv.store.channel, %length
%dkv.store.local = add i32 %dkv.store.channel.base, %dkv.store.key
%dkv.store.key.row = add i32 %dkv.input.row, %from
%dkv.store.key.index = add i32 %dkv.store.key.row, %dkv.store.local
%dkv.store.key.ptr = getelementptr inbounds float, ptr addrspace(1) %previous, i32 %dkv.store.key.index
%dkv.store.key.shared.index = add i32 %dkv.key.gradient.base.shared, %dkv.store.p
%dkv.store.key.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.store.key.shared.index
%dkv.store.key.value = load float, ptr addrspace(3) %dkv.store.key.shared.ptr, align 4
store float %dkv.store.key.value, ptr addrspace(1) %dkv.store.key.ptr, align 4
br label %dkv.store.next
dkv.store.value.step:
%dkv.store.value.channel = add i32 %dkv.store.value.head.start, %dkv.store.channel.local
%dkv.store.value.channel.base = mul i32 %dkv.store.value.channel, %length
%dkv.store.value.local = add i32 %dkv.store.value.channel.base, %dkv.store.key
%dkv.store.value.row = add i32 %dkv.input.row, %value.plane.base
%dkv.store.value.index = add i32 %dkv.store.value.row, %dkv.store.value.local
%dkv.store.value.ptr = getelementptr inbounds float, ptr addrspace(1) %previous, i32 %dkv.store.value.index
%dkv.store.value.shared.index = add i32 %dkv.value.gradient.base.shared, %dkv.store.p
%dkv.store.value.shared.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %dkv.store.value.shared.index
%dkv.store.value.value = load float, ptr addrspace(3) %dkv.store.value.shared.ptr, align 4
store float %dkv.store.value.value, ptr addrspace(1) %dkv.store.value.ptr, align 4
br label %dkv.store.next
dkv.store.next:
%dkv.store.p.next = add i32 %dkv.store.p, %block
br label %dkv.store.loop
dkv.store.done:
call void @llvm.nvvm.barrier0()
br label %dkv.job.finish
dkv.job.finish:
%dkv.job.next = add i32 %dkv.job, %groups
br label %dkv.job.loop
exit:
ret void
}
define internal void @scan_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output,
	ptr addrspace(1) %context, i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels, i32 %time.begin, i32 %time.span, i32 %gates,
	i1 %has.bias, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode, i1 %coded, i32 %cell.activation ) #3 { entry: %tid = call i32 @global_id()
%time.limit = add i32 %time.begin, %time.span
%weight.packed = icmp ne i32 %decode, 0
%in.channels.wide = zext i32 %in.channels to i64 %out.channels.wide = zext i32 %out.channels to i64 %length.wide = zext i32 %length to i64 %rows.wide = zext i32 %rows to i64 %weight.base.wide = add i64 %weight.base, 0
%in.elements = mul i32 %in.channels, %length %in.elements.wide = mul i64 %in.channels.wide, %length.wide %out.elements = mul i32 %out.channels, %length %out.elements.wide = mul i64 %out.channels.wide, %length.wide %input.matrix = mul i32 %in.channels, %out.channels %input.matrix.wide = mul i64 %in.channels.wide, %out.channels.wide
%state.matrix = mul i32 %out.channels, %out.channels %state.matrix.wide = mul i64 %out.channels.wide, %out.channels.wide %matrix.span = add i32 %input.matrix, %state.matrix %matrix.span.wide = add i64 %input.matrix.wide, %state.matrix.wide
	%bias.span = select i1 %has.bias, i32 %out.channels, i32 0
	%bias.span.wide = select i1 %has.bias, i64 %out.channels.wide, i64 0
	%gate.stride = add i32 %matrix.span, %bias.span %gate.stride.wide = add i64 %matrix.span.wide, %bias.span.wide %gate.batch = mul i32 %rows, %out.elements %gate.batch.wide = mul i64 %rows.wide, %out.elements.wide
br label %precompute.loop precompute.loop:
%precompute.gate = phi i32 [ 0, %entry ], [ %precompute.next, %precompute.step ]
%precompute.more = icmp ult i32 %precompute.gate, %gates
br i1 %precompute.more, label %precompute.step, label %precompute.done precompute.step:
%precompute.gate.wide = zext i32 %precompute.gate to i64 %precompute.weight.offset = mul i64 %precompute.gate.wide, %gate.stride.wide
%precompute.dense = getelementptr i8, ptr addrspace(1) %weights, i64 %precompute.weight.offset
%precompute.weights = select i1 %weight.packed, ptr addrspace(1) %weights, ptr addrspace(1) %precompute.dense
%precompute.base = add i64 %weight.base, %precompute.weight.offset
%precompute.context.offset = mul i64 %precompute.gate.wide, %gate.batch.wide
%precompute.context = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %precompute.context.offset
call void @contraction_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %precompute.weights,
ptr addrspace(1) %precompute.context, ptr addrspace(1) %input,
i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels,
i32 %length, i32 %time.begin, i32 %time.span, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false,
i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %precompute.base, i32 %decode )
%precompute.next = add i32 %precompute.gate, 1 br label %precompute.loop precompute.done:
call void @grid_barrier(i32 %threads) br label %row.loop row.loop:
%row = phi i32 [ %tid, %precompute.done ], [ %row.next, %time.done ] %row.wide = zext i32 %row to i64 %row.more = icmp ult i32 %row, %rows
br i1 %row.more, label %time.loop, label %exit time.loop: %time = phi i32 [ %time.begin, %row.loop ], [ %time.next, %output.done ]
%time.wide = zext i32 %time to i64 %previous.exists = icmp ne i32 %time, 0 %output.row.base = mul i64 %row.wide, %out.elements.wide
%time.more = icmp ult i32 %time, %time.limit br i1 %time.more, label %gate.loop, label %time.done gate.loop:
%gate = phi i32 [ 0, %time.loop ], [ %gate.next, %hidden.done ] %gate.wide = zext i32 %gate to i64 %gate.more = icmp ult i32 %gate, %gates
br i1 %gate.more, label %hidden.loop, label %output.loop hidden.loop:
%hidden = phi i32 [ 0, %gate.loop ], [ %hidden.next, %gate.store ] %gate.weight.base = mul i64 %gate.wide, %gate.stride.wide
%hidden.more = icmp ult i32 %hidden, %out.channels br i1 %hidden.more, label %input.load, label %hidden.done
input.load: %hidden.wide = zext i32 %hidden to i64 %input.gate.base = mul i64 %gate.wide, %gate.batch.wide %input.hidden.base = mul i64 %hidden.wide, %length.wide
%input.local = add i64 %input.hidden.base, %time.wide %input.row.local = add i64 %output.row.base, %input.local
%input.index = add i64 %input.gate.base, %input.row.local
%input.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %input.index
%input.sum = load i8, ptr addrspace(1) %input.ptr, align 1 br label %state.sum.loop state.sum.loop:
%state.channel = phi i32 [ 0, %input.load ], [ %state.next, %state.weight.ready ]
%state.sum = phi i8 [ %input.sum, %input.load ], [ %state.sum.next, %state.weight.ready ]
%state.more = icmp ult i32 %state.channel, %out.channels br i1 %state.more, label %state.sum.step, label %gate.activate
state.sum.step: %previous.time = sub i32 %time, 1 %previous.safe = select i1 %previous.exists, i32 %previous.time, i32 0 %state.channel.wide = zext i32 %state.channel to i64 %previous.safe.wide = zext i32 %previous.safe to i64
%state.channel.base = mul i64 %state.channel.wide, %length.wide %previous.local = add i64 %state.channel.base, %previous.safe.wide
%previous.index = add i64 %output.row.base, %previous.local
%previous.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %previous.index
%previous.loaded = load i8, ptr addrspace(1) %previous.ptr, align 1
%previous = select i1 %previous.exists, i8 %previous.loaded, i8 0 %candidate.gate = icmp eq i32 %gate, 2
%gru = icmp eq i32 %gates, 3 %reset.candidate = and i1 %gru, %candidate.gate
%reset.channel.base = mul i64 %state.channel.wide, %length.wide %reset.local = add i64 %reset.channel.base, %time.wide
%reset.row.index = add i64 %output.row.base, %reset.local %reset.base = add i64 %gate.batch.wide, %reset.row.index
%reset.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %reset.base
%reset = load i8, ptr addrspace(1) %reset.ptr, align 1 %reset.state = call i8 @recipe.mul(i8 %reset, i8 %previous)
%state.value = select i1 %reset.candidate, i8 %reset.state, i8 %previous
%state.weight.base = add i64 %gate.weight.base, %input.matrix.wide %state.weight.row = mul i64 %state.channel.wide, %out.channels.wide
%hidden.wide.state = zext i32 %hidden to i64 %state.weight.local = add i64 %state.weight.row, %hidden.wide.state
%state.weight.index = add i64 %state.weight.base, %state.weight.local
br i1 %weight.packed, label %state.weight.packed, label %state.weight.direct
state.weight.direct:
%state.weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %state.weight.index
%state.weight.loaded = load i8, ptr addrspace(1) %state.weight.ptr, align 1
br label %state.weight.ready
state.weight.packed:
%state.weight.decode.index = add i64 %weight.base.wide, %state.weight.index
%state.weight.decoded = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %state.weight.decode.index, i32 %decode)
br label %state.weight.ready
state.weight.ready:
%state.weight = phi i8 [ %state.weight.loaded, %state.weight.direct ], [ %state.weight.decoded, %state.weight.packed ]
%state.product = call i8 @recipe.mul(i8 %state.value, i8 %state.weight) %state.sum.next = call i8 @recipe.add(i8 %state.sum, i8 %state.product)
%state.next = add nuw i32 %state.channel, 1 br label %state.sum.loop gate.activate:
	%bias.base = add i64 %gate.weight.base, %matrix.span.wide %bias.hidden = zext i32 %hidden to i64 %bias.index = add i64 %bias.base, %bias.hidden
br i1 %weight.packed, label %gate.bias.packed, label %gate.bias.direct
gate.bias.direct:
%bias.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %bias.index
%bias.loaded = load i8, ptr addrspace(1) %bias.ptr, align 1
br label %gate.bias.ready
gate.bias.packed:
%bias.decode.index = add i64 %weight.base.wide, %bias.index
%bias.decoded = call i8 @recipe.model.decode(ptr addrspace(1) %weights, i64 %bias.decode.index, i32 %decode)
br label %gate.bias.ready
gate.bias.ready:
	%bias = phi i8 [ %bias.loaded, %gate.bias.direct ], [ %bias.decoded, %gate.bias.packed ]
	%bias.value = select i1 %has.bias, i8 %bias, i8 0
	%linear = call i8 @recipe.add(i8 %state.sum, i8 %bias.value)
%rnn = icmp eq i32 %gates, 1 %last.gate = sub i32 %gates, 1 %candidate = icmp eq i32 %gate, %last.gate
%use.tanh = or i1 %rnn, %candidate %tanh.value = call i8 @recipe.tanh(i8 %linear)
%sigmoid.value = call i8 @sigmoid(i8 %linear)
%builtin.value = select i1 %use.tanh, i8 %tanh.value, i8 %sigmoid.value
%coded.relu = icmp eq i32 %cell.activation, 1 %coded.tanh = icmp eq i32 %cell.activation, 2 %coded.sigmoid = icmp eq i32 %cell.activation, 3
%coded.positive = call i1 @recipe.ogt(i8 %linear, i8 0) %coded.relu.value = select i1 %coded.positive, i8 %linear, i8 0
%coded.a = select i1 %coded.relu, i8 %coded.relu.value, i8 %linear
%coded.b = select i1 %coded.tanh, i8 %tanh.value, i8 %coded.a
%coded.value = select i1 %coded.sigmoid, i8 %sigmoid.value, i8 %coded.b
%gate.value = select i1 %coded, i8 %coded.value, i8 %builtin.value br label %gate.store gate.store:
%gate.context.base = mul i64 %gate.wide, %gate.batch.wide %gate.hidden.base = mul i64 %hidden.wide, %length.wide
%gate.local = add i64 %gate.hidden.base, %time.wide %gate.row.local = add i64 %output.row.base, %gate.local
%gate.index = add i64 %gate.context.base, %gate.row.local
%gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %gate.index
store i8 %gate.value, ptr addrspace(1) %gate.ptr, align 1 %hidden.next = add nuw i32 %hidden, 1
br label %hidden.loop hidden.done: %gate.next = add nuw i32 %gate, 1 br label %gate.loop output.loop:
%output.hidden = phi i32 [ 0, %gate.loop ], [ %output.next, %output.store ]
%output.more = icmp ult i32 %output.hidden, %out.channels br i1 %output.more, label %output.step, label %output.done
output.step: %output.hidden.wide = zext i32 %output.hidden to i64 %output.hidden.base = mul i64 %output.hidden.wide, %length.wide %output.local = add i64 %output.hidden.base, %time.wide
%output.index = add i64 %output.row.base, %output.local
%gate0.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %output.index
%gate0 = load i8, ptr addrspace(1) %gate0.ptr, align 1
%is.gru = icmp eq i32 %gates, 3 %is.lstm = icmp eq i32 %gates, 4
%gate1.raw = add i64 %gate.batch.wide, %output.index %gate1.index = select i1 %is.lstm, i64 %gate1.raw, i64 %output.index
%gate1.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %gate1.index
%gate1 = load i8, ptr addrspace(1) %gate1.ptr, align 1 %gate2.base = mul i64 %gate.batch.wide, 2
%gate2.raw = add i64 %gate2.base, %output.index %gate2.valid = or i1 %is.gru, %is.lstm
%gate2.index = select i1 %gate2.valid, i64 %gate2.raw, i64 %output.index
%gate2.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %gate2.index
%gate2 = load i8, ptr addrspace(1) %gate2.ptr, align 1 %gate3.base = mul i64 %gate.batch.wide, 3
%gate3.raw = add i64 %gate3.base, %output.index %gate3.index = select i1 %is.lstm, i64 %gate3.raw, i64 %output.index
%gate3.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %gate3.index
%gate3 = load i8, ptr addrspace(1) %gate3.ptr, align 1 %output.previous.time = sub i32 %time, 1
%output.previous.safe = select i1 %previous.exists, i32 %output.previous.time, i32 0 %output.previous.safe.wide = zext i32 %output.previous.safe to i64
%output.previous.local = add i64 %output.hidden.base, %output.previous.safe.wide
%output.previous.index = add i64 %output.row.base, %output.previous.local
%output.previous.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.previous.index
%output.previous.loaded = load i8, ptr addrspace(1) %output.previous.ptr, align 1
%output.previous = select i1 %previous.exists, i8 %output.previous.loaded, i8 0
%one.update = call i8 @recipe.sub(i8 56, i8 %gate0) %gru.old = call i8 @recipe.mul(i8 %gate0, i8 %output.previous)
%gru.new = call i8 @recipe.mul(i8 %one.update, i8 %gate2) %gru.value = call i8 @recipe.add(i8 %gru.old, i8 %gru.new)
%gates.wide = zext i32 %gates to i64 %cell.base = mul i64 %gate.batch.wide, %gates.wide %cell.index = add i64 %cell.base, %output.index
%cell.previous.index = add i64 %cell.base, %output.previous.index
%cell.previous.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %cell.previous.index
%cell.previous.loaded = load i8, ptr addrspace(1) %cell.previous.ptr, align 1
%cell.previous = select i1 %previous.exists, i8 %cell.previous.loaded, i8 0
%cell.old = call i8 @recipe.mul(i8 %gate1, i8 %cell.previous) %cell.new = call i8 @recipe.mul(i8 %gate0, i8 %gate3)
%cell = call i8 @recipe.add(i8 %cell.old, i8 %cell.new)
%cell.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %cell.index
store i8 %cell, ptr addrspace(1) %cell.ptr, align 1 %cell.tanh = call i8 @recipe.tanh(i8 %cell)
%lstm.value = call i8 @recipe.mul(i8 %gate2, i8 %cell.tanh)
%rnn.or.gru = select i1 %is.gru, i8 %gru.value, i8 %gate0
%output.value = select i1 %is.lstm, i8 %lstm.value, i8 %rnn.or.gru br label %output.store output.store:
%output.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %output.index
store i8 %output.value, ptr addrspace(1) %output.ptr, align 1 %output.next = add nuw i32 %output.hidden, 1
br label %output.loop output.done: %time.next = add nuw i32 %time, 1 br label %time.loop time.done:
%row.next = add i32 %row, %threads br label %row.loop exit: ret void }
define internal void @contraction_reverse_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %delta, ptr addrspace(1) %previous, ptr addrspace(1) %gradient, i1 %write.input, i1 %has.bias, i1 %relu, i1 %matrix.gradient,
i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %kernel, i32 %offset,
i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) RECIPE_CONTRACTION_BODY { entry:
%sums = alloca [RECIPE_REGISTER_COUNT x float], align 4
%bias.sums = alloca [RECIPE_REGISTER_N x float], align 4
%state.zero = call float @recipe.state.from.u1(i1 false) %lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() %group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() %block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %groups = udiv i32 %threads, %block
%in.channels.wide = zext i32 %in.channels to i64 %in.length.wide = zext i32 %in.length to i64 %out.channels.wide = zext i32 %out.channels to i64 %out.length.wide = zext i32 %out.length to i64 %rows.wide = zext i32 %rows to i64 %offset.wide = zext i32 %offset to i64
%in.elements = mul i32 %in.channels, %in.length %in.elements.wide = mul i64 %in.channels.wide, %in.length.wide %out.elements = mul i32 %out.channels, %out.length %out.elements.wide = mul i64 %out.channels.wide, %out.length.wide %is.conv = icmp ne i32 %kernel, 0
%span = select i1 %is.conv, i32 %kernel, i32 1 %window = mul i32 %in.channels, %span %window.wide = zext i32 %window to i64
%gradient.r.total = mul i32 %rows, %out.length
%gradient.matrix.values = mul i32 %out.channels, %window
%gradient.bias.values = select i1 %has.bias, i32 %out.channels, i32 0
%gradient.values = add i32 %gradient.matrix.values, %gradient.bias.values
; Split-K scratch rows are written by different workgroups. Pad each row so no
; two rows share a machine word and a partial store cannot lose a neighbour. The
; base is aligned by the host, so row zero starts on the same boundary.
%gradient.stride.raw = add i32 %gradient.values, RECIPE_SCRATCH_ROW_MASK
%gradient.stride = and i32 %gradient.stride.raw, RECIPE_SCRATCH_ROW_CLEAR
%gradient.scratch = getelementptr inbounds float, ptr addrspace(1) %gradient, i32 RECIPE_GRADIENT_SCRATCH_BASE
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
%gradient.store.offset = add i32 %gradient.destination.base, %gradient.store.row %gradient.store.offset.wide = zext i32 %gradient.store.offset to i64
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
%gradient.m.lanes.adjusted = add i32 %gradient.m.count, RECIPE_REGISTER_M %gradient.m.lanes.numerator = sub i32 %gradient.m.lanes.adjusted, 1 %gradient.m.lanes = udiv i32 %gradient.m.lanes.numerator, RECIPE_REGISTER_M %gradient.n.lanes.adjusted = add i32 %gradient.n.count, RECIPE_REGISTER_N %gradient.n.lanes.numerator = sub i32 %gradient.n.lanes.adjusted, 1 %gradient.n.lanes = udiv i32 %gradient.n.lanes.numerator, RECIPE_REGISTER_N
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
%gradient.output.m.base = mul i32 %gradient.lane.m, RECIPE_REGISTER_M %gradient.output.n.base = mul i32 %gradient.lane.n, RECIPE_REGISTER_N
%gradient.bias.first = icmp eq i32 %gradient.m.base, 0
%gradient.bias.channel = icmp ult i32 %lid, %gradient.n.count
%gradient.bias.owner = and i1 %gradient.bias.first, %gradient.bias.channel
%gradient.bias.enable = and i1 %has.bias, %gradient.bias.owner
br label %gradient.sum.init.loop gradient.sum.init.loop:
%gradient.sum.init = phi i32 [ 0, %gradient.job.step ], [ %gradient.sum.init.next, %gradient.sum.init.step ] %gradient.sum.init.more = icmp ult i32 %gradient.sum.init, RECIPE_REGISTER_COUNT br i1 %gradient.sum.init.more, label %gradient.sum.init.step, label %gradient.tile.loop
gradient.sum.init.step: %gradient.sum.init.ptr = getelementptr [RECIPE_REGISTER_COUNT x float], ptr %sums, i32 0, i32 %gradient.sum.init store float %state.zero, ptr %gradient.sum.init.ptr, align 4 %gradient.sum.init.next = add i32 %gradient.sum.init, 1 br label %gradient.sum.init.loop gradient.tile.loop:
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
%gradient.a.fragment.remainder = urem i32 %gradient.m.count, RECIPE_FRAGMENT_K
%gradient.a.fragment.full = icmp eq i32 %gradient.a.fragment.remainder, 0
%gradient.a.vector = and i1 %gradient.a.contiguous, %gradient.a.fragment.full
%gradient.a.width = select i1 %gradient.a.vector, i32 RECIPE_FRAGMENT_K, i32 1
%gradient.a.columns = udiv i32 %gradient.m.count, %gradient.a.width
%gradient.b.unit = icmp eq i32 %out.length, 1
%gradient.b.fragment.remainder = urem i32 %gradient.n.count, RECIPE_FRAGMENT_K
%gradient.b.fragment.full = icmp eq i32 %gradient.b.fragment.remainder, 0
%gradient.b.vector = and i1 %gradient.b.unit, %gradient.b.fragment.full
%gradient.b.width = select i1 %gradient.b.vector, i32 RECIPE_FRAGMENT_K, i32 1
%gradient.b.columns = udiv i32 %gradient.n.count, %gradient.b.width
%gradient.a.count = mul i32 %gradient.a.columns, %gradient.r.count %gradient.b.count = mul i32 %gradient.b.columns, %gradient.r.count %gradient.load.count = add i32 %gradient.a.count, %gradient.b.count br label %gradient.load.loop gradient.load.loop:
%gradient.load = phi i32 [ %lid, %gradient.load.generic.entry ], [ %gradient.load.next, %gradient.load.advance ] %gradient.load.more = icmp ult i32 %gradient.load, %gradient.load.count br i1 %gradient.load.more, label %gradient.load.classify, label %gradient.load.done
gradient.load.classify: %gradient.load.a = icmp ult i32 %gradient.load, %gradient.a.count br i1 %gradient.load.a, label %gradient.load.a.step, label %gradient.load.b.step
gradient.load.a.step: %gradient.a.r = udiv i32 %gradient.load, %gradient.a.columns %gradient.a.column = urem i32 %gradient.load, %gradient.a.columns %gradient.a.m = mul i32 %gradient.a.column, %gradient.a.width %gradient.a.global = add i32 %gradient.r.base, %gradient.a.r
%gradient.a.row = udiv i32 %gradient.a.global, %out.length %gradient.a.position = urem i32 %gradient.a.global, %out.length %gradient.a.row.wide = zext i32 %gradient.a.row to i64 %gradient.a.position.wide = zext i32 %gradient.a.position to i64 %gradient.a.row.base = mul i64 %gradient.a.row.wide, %in.elements.wide %gradient.a.term = add i32 %gradient.m.base, %gradient.a.m %gradient.a.term.wide = zext i32 %gradient.a.term to i64
%gradient.a.tile.index = call i32 @contraction_vector_a_index(i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br i1 %gradient.a.vector, label %gradient.load.a.vector, label %gradient.load.a.scalar
gradient.load.a.vector:
%gradient.a.vector.index = add i64 %gradient.a.row.base, %gradient.a.term.wide
%gradient.a.vector.source = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %gradient.a.vector.index
%gradient.a.vector.value = load <RECIPE_FRAGMENT_K x i8>, ptr addrspace(1) %gradient.a.vector.source, align 1
call void @contraction_stage_a_columns(<RECIPE_FRAGMENT_K x i8> %gradient.a.vector.value, i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.a.scalar:
%gradient.a.value = call i8 @contraction_input( ptr addrspace(1) %input, i64 %gradient.a.row.base, i32 %gradient.a.position, i32 %gradient.a.term, i32 %span, i32 %in.length, i1 %is.conv )
br label %gradient.load.store
gradient.load.b.step: %gradient.b.local = sub i32 %gradient.load, %gradient.a.count %gradient.b.r = udiv i32 %gradient.b.local, %gradient.b.columns %gradient.b.column = urem i32 %gradient.b.local, %gradient.b.columns %gradient.b.n = mul i32 %gradient.b.column, %gradient.b.width %gradient.b.global = add i32 %gradient.r.base, %gradient.b.r
%gradient.b.row = udiv i32 %gradient.b.global, %out.length %gradient.b.position = urem i32 %gradient.b.global, %out.length %gradient.b.filter = add i32 %gradient.n.base, %gradient.b.n
%gradient.b.row.wide = zext i32 %gradient.b.row to i64 %gradient.b.position.wide = zext i32 %gradient.b.position to i64 %gradient.b.filter.wide = zext i32 %gradient.b.filter to i64 %gradient.b.row.base = mul i64 %gradient.b.row.wide, %out.elements.wide %gradient.b.filter.base = mul i64 %gradient.b.filter.wide, %out.length.wide %gradient.b.local.index = add i64 %gradient.b.filter.base, %gradient.b.position.wide %gradient.b.index = add i64 %gradient.b.row.base, %gradient.b.local.index
%gradient.b.model.elements = mul i32 %gradient.tile.m, %gradient.tile.k
%gradient.b.tile.base = call i32 @contraction_state_after_model(i32 %gradient.b.model.elements)
%gradient.b.tile.local = call i32 @contraction_vector_b_index(i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.n, i32 %gradient.tile.k) %gradient.b.tile.index = add i32 %gradient.b.tile.base, %gradient.b.tile.local
br i1 %gradient.b.vector, label %gradient.load.b.vector, label %gradient.load.b.scalar
gradient.load.b.vector:
%gradient.b.vector.value = call <RECIPE_FRAGMENT_K x float> @contraction_delta_vector16_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %gradient.b.index, i1 %relu)
call void @contraction_stage_b_fragment_state(<RECIPE_FRAGMENT_K x float> %gradient.b.vector.value, i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.b.scalar:
%gradient.b.value = call float @contraction_delta_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %gradient.b.index, i1 %relu)
%gradient.load.b.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %gradient.b.tile.index store float %gradient.b.value, ptr addrspace(3) %gradient.load.b.ptr, align 4
br label %gradient.load.advance
gradient.load.store:
%gradient.load.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %gradient.a.tile.index store i8 %gradient.a.value, ptr addrspace(3) %gradient.load.ptr, align 1
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
call void @contraction_zero_edges_bs(i32 %gradient.m.count, i32 %gradient.n.count, i32 %gradient.r.count, i32 %lid, i32 %block, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.load.ready
gradient.load.ready:
call void @llvm.nvvm.barrier0()
br label %gradient.scalar.ready
gradient.scalar.ready:
call void @contraction_bias_accumulate_state(ptr %bias.sums, ptr addrspace(1) %gradient.destination, i1 %gradient.bias.enable, i1 %gradient.r.first.tile, i1 %gradient.r.last.tile, i32 %lid, i32 %block, i32 %gradient.n.base, i32 %gradient.n.count, i32 %gradient.r.count, i32 %out.channels, i32 %window, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %gradient.store.offset)
call void @contraction_vector_accumulate_bs(ptr %sums, i1 %gradient.lane.active, i1 %gradient.method.store, i32 %lid, i32 %gradient.lane.k, i32 %gradient.k.lanes, i32 %gradient.output.lane, i32 %gradient.output.lanes, i32 %gradient.output.m.base, i32 %gradient.output.n.base, i32 %gradient.m.count, i32 %gradient.n.count, i32 %gradient.r.count, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.product.done
gradient.product.done:
call void @llvm.nvvm.barrier0()
br i1 %gradient.r.more, label %gradient.tile.done, label %gradient.store.loop gradient.tile.done: br label %gradient.tile.loop
gradient.store.loop:
%gradient.store.register = phi i32 [ 0, %gradient.product.done ], [ %gradient.store.register.next, %gradient.store.next ] %gradient.store.more = icmp ult i32 %gradient.store.register, RECIPE_REGISTER_COUNT br i1 %gradient.store.more, label %gradient.store.check, label %gradient.bias.store.check
gradient.store.check: %gradient.store.register.m = urem i32 %gradient.store.register, RECIPE_REGISTER_M %gradient.store.register.n = udiv i32 %gradient.store.register, RECIPE_REGISTER_M %gradient.store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.register.valid = call i1 @contraction_output_register_valid(i32 %gradient.store.register)
%gradient.store.output.m.valid = icmp ult i32 %gradient.store.output.m.raw, %gradient.m.count %gradient.store.output.n.valid = icmp ult i32 %gradient.store.output.n.raw, %gradient.n.count %gradient.store.output.valid = and i1 %gradient.store.output.m.valid, %gradient.store.output.n.valid %gradient.store.lane.active = and i1 %gradient.method.store, %gradient.store.output.valid %gradient.store.active = and i1 %gradient.store.lane.active, %gradient.store.register.valid br i1 %gradient.store.active, label %gradient.store, label %gradient.store.next
gradient.store: %gradient.store.filter = add i32 %gradient.n.base, %gradient.store.output.n.raw %gradient.store.term = add i32 %gradient.m.base, %gradient.store.output.m.raw %gradient.store.filter.wide = zext i32 %gradient.store.filter to i64 %gradient.store.term.wide = zext i32 %gradient.store.term to i64 %gradient.store.filter.base = mul i64 %gradient.store.filter.wide, %window.wide %gradient.store.local = add i64 %gradient.store.filter.base, %gradient.store.term.wide %gradient.store.index = add i64 %gradient.store.offset.wide, %gradient.store.local
%gradient.store.ptr = getelementptr inbounds float, ptr addrspace(1) %gradient.destination, i64 %gradient.store.index %gradient.store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x float], ptr %sums, i32 0, i32 %gradient.store.register %gradient.store.sum.wide = load float, ptr %gradient.store.sum.ptr, align 4 store float %gradient.store.sum.wide, ptr addrspace(1) %gradient.store.ptr, align 4
br label %gradient.store.next
gradient.store.next: %gradient.store.register.next = add i32 %gradient.store.register, 1 br label %gradient.store.loop gradient.bias.store.check:
br label %gradient.job.done
gradient.job.done: %gradient.task.next = add i32 %gradient.task, %groups br label %gradient.job.loop
gradient.finish:
br i1 %gradient.direct, label %previous.check, label %gradient.reduce.entry
gradient.reduce.entry:
call void @grid_barrier(i32 %threads)
call void @reduce_rows_state(ptr addrspace(1) %gradient.scratch, ptr addrspace(1) %gradient, i32 %gradient.splits, i32 %gradient.values, i32 %gradient.stride, i32 0, i32 %offset, i32 %threads)
br label %previous.check
previous.check: br i1 %write.input, label %previous.entry, label %exit previous.entry:
%previous.span.wide = zext i32 %span to i64
%previous.m.total = mul i32 %rows, %in.length %previous.r.total = mul i32 %out.channels, %span
%previous.m.short = icmp ult i32 %previous.tile.m, %previous.m.total %previous.m.tile = select i1 %previous.m.short, i32 %previous.tile.m, i32 %previous.m.total %previous.n.short = icmp ult i32 %previous.tile.n, %in.channels %previous.n.tile = select i1 %previous.n.short, i32 %previous.tile.n, i32 %in.channels
%previous.k.short = icmp ult i32 %previous.tile.k, %previous.r.total %previous.k.tile = select i1 %previous.k.short, i32 %previous.tile.k, i32 %previous.r.total
%previous.m.adjusted = add i32 %previous.m.total, %previous.m.tile %previous.m.numerator = sub i32 %previous.m.adjusted, 1 %previous.m.tiles = udiv i32 %previous.m.numerator, %previous.m.tile %previous.n.adjusted = add i32 %in.channels, %previous.n.tile %previous.n.numerator = sub i32 %previous.n.adjusted, 1 %previous.n.tiles = udiv i32 %previous.n.numerator, %previous.n.tile
%previous.jobs = mul i32 %previous.m.tiles, %previous.n.tiles br label %previous.job.loop previous.job.loop:
%previous.job = phi i32 [ %group, %previous.entry ], [ %previous.job.next, %previous.job.done ] %previous.job.more = icmp ult i32 %previous.job, %previous.jobs br i1 %previous.job.more, label %previous.job.step, label %exit
previous.job.step: %previous.m.group.short = icmp ult i32 %previous.m.tiles, 4 %previous.m.group.limit = select i1 %previous.m.group.short, i32 %previous.m.tiles, i32 4 %previous.group.width = mul i32 %previous.m.group.limit, %previous.n.tiles %previous.group.index = udiv i32 %previous.job, %previous.group.width %previous.m.group.base = mul i32 %previous.group.index, %previous.m.group.limit %previous.m.group.remaining = sub i32 %previous.m.tiles, %previous.m.group.base %previous.m.group.tail = icmp ult i32 %previous.m.group.remaining, %previous.m.group.limit %previous.m.group.count = select i1 %previous.m.group.tail, i32 %previous.m.group.remaining, i32 %previous.m.group.limit %previous.group.local = urem i32 %previous.job, %previous.group.width %previous.m.group.local = urem i32 %previous.group.local, %previous.m.group.count %previous.m.index = add i32 %previous.m.group.base, %previous.m.group.local %previous.n.index = udiv i32 %previous.group.local, %previous.m.group.count %previous.m.base = mul i32 %previous.m.index, %previous.m.tile %previous.n.base = mul i32 %previous.n.index, %previous.n.tile
%previous.m.remaining = sub i32 %previous.m.total, %previous.m.base %previous.m.partial = icmp ult i32 %previous.m.remaining, %previous.m.tile %previous.m.count = select i1 %previous.m.partial, i32 %previous.m.remaining, i32 %previous.m.tile
%previous.n.remaining = sub i32 %in.channels, %previous.n.base %previous.n.partial = icmp ult i32 %previous.n.remaining, %previous.n.tile %previous.n.count = select i1 %previous.n.partial, i32 %previous.n.remaining, i32 %previous.n.tile
%previous.m.lanes.adjusted = add i32 %previous.m.count, RECIPE_REGISTER_M %previous.m.lanes.numerator = sub i32 %previous.m.lanes.adjusted, 1 %previous.m.lanes = udiv i32 %previous.m.lanes.numerator, RECIPE_REGISTER_M %previous.n.lanes.adjusted = add i32 %previous.n.count, RECIPE_REGISTER_N %previous.n.lanes.numerator = sub i32 %previous.n.lanes.adjusted, 1 %previous.n.lanes = udiv i32 %previous.n.lanes.numerator, RECIPE_REGISTER_N
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
%previous.output.m.base = mul i32 %previous.lane.m, RECIPE_REGISTER_M %previous.output.n.base = mul i32 %previous.lane.n, RECIPE_REGISTER_N br label %previous.sum.init.loop previous.sum.init.loop:
%previous.sum.init = phi i32 [ 0, %previous.job.step ], [ %previous.sum.init.next, %previous.sum.init.step ] %previous.sum.init.more = icmp ult i32 %previous.sum.init, RECIPE_REGISTER_COUNT br i1 %previous.sum.init.more, label %previous.sum.init.step, label %previous.tile.loop
previous.sum.init.step: %previous.sum.init.ptr = getelementptr [RECIPE_REGISTER_COUNT x float], ptr %sums, i32 0, i32 %previous.sum.init store float %state.zero, ptr %previous.sum.init.ptr, align 4 %previous.sum.init.next = add i32 %previous.sum.init, 1 br label %previous.sum.init.loop previous.tile.loop:
%previous.r.base = phi i32 [ 0, %previous.sum.init.loop ], [ %previous.r.next, %previous.tile.done ]
%previous.r.remaining = sub i32 %previous.r.total, %previous.r.base %previous.r.partial = icmp ult i32 %previous.r.remaining, %previous.k.tile %previous.r.count = select i1 %previous.r.partial, i32 %previous.r.remaining, i32 %previous.k.tile
%previous.a.project = icmp eq i32 %span, 1 %previous.a.unit = icmp eq i32 %out.length, 1 %previous.a.contiguous = and i1 %previous.a.project, %previous.a.unit
%previous.a.fragment.remainder = urem i32 %previous.r.count, RECIPE_FRAGMENT_K %previous.a.fragment.full = icmp eq i32 %previous.a.fragment.remainder, 0 %previous.a.vector = and i1 %previous.a.contiguous, %previous.a.fragment.full
%previous.a.width = select i1 %previous.a.vector, i32 RECIPE_FRAGMENT_K, i32 1 %previous.a.columns = udiv i32 %previous.r.count, %previous.a.width
%previous.b.fragment.remainder = urem i32 %previous.n.count, RECIPE_FRAGMENT_K %previous.b.fragment.full = icmp eq i32 %previous.b.fragment.remainder, 0 %previous.b.vector = and i1 %previous.a.project, %previous.b.fragment.full
%previous.b.width = select i1 %previous.b.vector, i32 RECIPE_FRAGMENT_K, i32 1 %previous.b.columns = udiv i32 %previous.n.count, %previous.b.width
%previous.a.count = mul i32 %previous.m.count, %previous.a.columns %previous.b.count = mul i32 %previous.r.count, %previous.b.columns %previous.load.count = add i32 %previous.a.count, %previous.b.count br label %previous.load.loop previous.load.loop:
%previous.load = phi i32 [ %lid, %previous.tile.loop ], [ %previous.load.next, %previous.load.advance ] %previous.load.more = icmp ult i32 %previous.load, %previous.load.count br i1 %previous.load.more, label %previous.load.classify, label %previous.load.done
previous.load.classify: %previous.load.a = icmp ult i32 %previous.load, %previous.a.count br i1 %previous.load.a, label %previous.load.a.step, label %previous.load.b.step
previous.load.a.step: %previous.a.m = udiv i32 %previous.load, %previous.a.columns %previous.a.column = urem i32 %previous.load, %previous.a.columns %previous.a.r = mul i32 %previous.a.column, %previous.a.width %previous.a.term = add i32 %previous.r.base, %previous.a.r %previous.a.term.wide = zext i32 %previous.a.term to i64
%previous.a.filter = udiv i32 %previous.a.term, %span %previous.a.kernel = urem i32 %previous.a.term, %span %previous.a.global = add i32 %previous.m.base, %previous.a.m %previous.a.row = udiv i32 %previous.a.global, %in.length %previous.a.position = urem i32 %previous.a.global, %in.length
%previous.a.low = icmp uge i32 %previous.a.position, %previous.a.kernel %previous.a.position.raw = sub i32 %previous.a.position, %previous.a.kernel %previous.a.high = icmp ult i32 %previous.a.position.raw, %out.length %previous.a.valid = and i1 %previous.a.low, %previous.a.high
%previous.a.position.safe = select i1 %previous.a.valid, i32 %previous.a.position.raw, i32 0 %previous.a.row.wide = zext i32 %previous.a.row to i64 %previous.a.filter.wide = zext i32 %previous.a.filter to i64 %previous.a.position.safe.wide = zext i32 %previous.a.position.safe to i64 %previous.a.row.base = mul i64 %previous.a.row.wide, %out.elements.wide %previous.a.filter.base = mul i64 %previous.a.filter.wide, %out.length.wide
%previous.a.local = add i64 %previous.a.filter.base, %previous.a.position.safe.wide %previous.a.index = add i64 %previous.a.row.base, %previous.a.local %previous.a.tile.index = call i32 @contraction_vector_a_index(i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br i1 %previous.a.vector, label %previous.load.a.vector, label %previous.load.a.scalar
previous.load.a.vector:
%previous.a.vector.delta = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %previous.a.index
%previous.a.vector.output = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %previous.a.index
%previous.a.vector.delta.value = load <RECIPE_FRAGMENT_K x float>, ptr addrspace(1) %previous.a.vector.delta, align 4
%previous.a.vector.output.value = load <RECIPE_FRAGMENT_K x i8>, ptr addrspace(1) %previous.a.vector.output, align 1
call void @contraction_stage_delta_a_fragment_state(<RECIPE_FRAGMENT_K x float> %previous.a.vector.delta.value, <RECIPE_FRAGMENT_K x i8> %previous.a.vector.output.value, i1 %relu, i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.a.scalar:
%previous.a.raw = call float @contraction_delta_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %previous.a.index, i1 %relu)
%previous.a.zero = call float @recipe.state.from.u1(i1 false)
%previous.a.value = select i1 %previous.a.valid, float %previous.a.raw, float %previous.a.zero
%previous.load.a.ptr = getelementptr float, ptr addrspace(3) @contraction_tile_f8, i32 %previous.a.tile.index store float %previous.a.value, ptr addrspace(3) %previous.load.a.ptr, align 4
br label %previous.load.advance
previous.load.b.step: %previous.b.local = sub i32 %previous.load, %previous.a.count %previous.b.r = udiv i32 %previous.b.local, %previous.b.columns %previous.b.column = urem i32 %previous.b.local, %previous.b.columns %previous.b.n = mul i32 %previous.b.column, %previous.b.width %previous.b.term = add i32 %previous.r.base, %previous.b.r
%previous.b.filter = udiv i32 %previous.b.term, %span %previous.b.kernel = urem i32 %previous.b.term, %span %previous.b.channel = add i32 %previous.n.base, %previous.b.n %previous.b.filter.wide = zext i32 %previous.b.filter to i64 %previous.b.channel.wide = zext i32 %previous.b.channel to i64 %previous.b.kernel.wide = zext i32 %previous.b.kernel to i64 %previous.b.filter.base = mul i64 %previous.b.filter.wide, %window.wide
%previous.b.channel.base = mul i64 %previous.b.channel.wide, %previous.span.wide %previous.b.channel.local = add i64 %previous.b.channel.base, %previous.b.kernel.wide %previous.b.index = add i64 %previous.b.filter.base, %previous.b.channel.local
%previous.a.state.elements = mul i32 %previous.tile.m, %previous.tile.k
%previous.b.tile.base = call i32 @contraction_model_after_state(i32 %previous.a.state.elements)
%previous.b.tile.local = call i32 @contraction_vector_b_index(i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.n, i32 %previous.tile.k) %previous.b.tile.index = add i32 %previous.b.tile.base, %previous.b.tile.local
br i1 %previous.b.vector, label %previous.load.b.vector, label %previous.load.b.scalar
previous.load.b.vector:
%previous.b.vector.source = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %previous.b.index
%previous.b.vector.value = load <RECIPE_FRAGMENT_K x i8>, ptr addrspace(1) %previous.b.vector.source, align 1
call void @contraction_stage_b_fragment_after_state(<RECIPE_FRAGMENT_K x i8> %previous.b.vector.value, i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.b.scalar:
%previous.b.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %previous.b.index
%previous.b.value = load i8, ptr addrspace(1) %previous.b.ptr, align 1
br label %previous.load.store
previous.load.store:
%previous.load.ptr = getelementptr [0 x i8], ptr addrspace(3) @contraction_tile_f8, i32 0, i32 %previous.b.tile.index store i8 %previous.b.value, ptr addrspace(3) %previous.load.ptr, align 1
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
call void @contraction_zero_edges_as(i32 %previous.m.count, i32 %previous.n.count, i32 %previous.r.count, i32 %lid, i32 %block, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k)
br label %previous.load.ready
previous.load.ready:
call void @llvm.nvvm.barrier0()
call void @contraction_vector_accumulate_as(ptr %sums, i1 %previous.lane.active, i1 %previous.method.store, i32 %lid, i32 %previous.lane.k, i32 %previous.k.lanes, i32 %previous.output.lane, i32 %previous.lanes, i32 %previous.output.m.base, i32 %previous.output.n.base, i32 %previous.m.count, i32 %previous.n.count, i32 %previous.r.count, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k) call void @llvm.nvvm.barrier0()
%previous.r.next = add i32 %previous.r.base, %previous.r.count %previous.r.more = icmp ult i32 %previous.r.next, %previous.r.total br i1 %previous.r.more, label %previous.tile.done, label %previous.store.loop previous.tile.done: br label %previous.tile.loop previous.store.loop:
%previous.store.register = phi i32 [ 0, %previous.load.ready ], [ %previous.store.register.next, %previous.store.next ] %previous.store.more = icmp ult i32 %previous.store.register, RECIPE_REGISTER_COUNT br i1 %previous.store.more, label %previous.store.check, label %previous.job.done
previous.store.check: %previous.store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.register.valid = call i1 @contraction_output_register_valid(i32 %previous.store.register)
%previous.store.output.m.valid = icmp ult i32 %previous.store.output.m.raw, %previous.m.count %previous.store.output.n.valid = icmp ult i32 %previous.store.output.n.raw, %previous.n.count %previous.store.output.valid = and i1 %previous.store.output.m.valid, %previous.store.output.n.valid %previous.lane.output.active = and i1 %previous.method.store, %previous.store.output.valid %previous.store.active = and i1 %previous.lane.output.active, %previous.store.register.valid br i1 %previous.store.active, label %previous.store, label %previous.store.next
previous.store: %previous.store.m.global = add i32 %previous.m.base, %previous.store.output.m.raw %previous.store.channel = add i32 %previous.n.base, %previous.store.output.n.raw %previous.store.m.global.wide = zext i32 %previous.store.m.global to i64 %previous.store.channel.wide = zext i32 %previous.store.channel to i64 %previous.store.row.wide = udiv i64 %previous.store.m.global.wide, %in.length.wide %previous.store.position.wide = urem i64 %previous.store.m.global.wide, %in.length.wide
%previous.store.row.base = mul i64 %previous.store.row.wide, %in.elements.wide %previous.store.channel.base = mul i64 %previous.store.channel.wide, %in.length.wide %previous.store.local = add i64 %previous.store.channel.base, %previous.store.position.wide %previous.store.index = add i64 %previous.store.row.base, %previous.store.local %previous.store.ptr = getelementptr inbounds float, ptr addrspace(1) %previous, i64 %previous.store.index
%previous.store.old = load float, ptr addrspace(1) %previous.store.ptr, align 4 %previous.store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x float], ptr %sums, i32 0, i32 %previous.store.register %previous.store.sum.wide = load float, ptr %previous.store.sum.ptr, align 4 %previous.store.value = call float @recipe.state.add(float %previous.store.old, float %previous.store.sum.wide) store float %previous.store.value, ptr addrspace(1) %previous.store.ptr, align 4 br label %previous.store.next
previous.store.next: %previous.store.register.next = add i32 %previous.store.register, 1 br label %previous.store.loop previous.job.done: %previous.job.next = add i32 %previous.job, %groups br label %previous.job.loop exit: ret void }
define internal void @scan_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output,
ptr addrspace(1) %context, ptr addrspace(1) %backward, ptr addrspace(1) %delta, ptr addrspace(1) %previous,
ptr addrspace(1) %gradient, i1 %write.input, i32 %rows, i32 %in.channels,
i32 %length, i32 %out.channels, i32 %gates, i1 %has.bias, i32 %parameters, i32 %offset,
i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads, i1 %coded, i32 %cell.activation ) #3 { entry:
	%tid = call i32 @global_id() %state.zero = call float @recipe.state.from.u1(i1 false) %state.one = call float @recipe.state.from.u1(i1 true) %scan.in.channels.wide = zext i32 %in.channels to i64 %scan.length.wide = zext i32 %length to i64 %scan.out.channels.wide = zext i32 %out.channels to i64 %scan.rows.wide = zext i32 %rows to i64 %scan.gates.wide = zext i32 %gates to i64 %scan.parameters.wide = zext i32 %parameters to i64 %in.elements = mul i32 %in.channels, %length %scan.in.elements.wide = mul i64 %scan.in.channels.wide, %scan.length.wide
%out.elements = mul i32 %out.channels, %length %scan.out.elements.wide = mul i64 %scan.out.channels.wide, %scan.length.wide %batch = mul i32 %rows, %out.elements %scan.batch.wide = mul i64 %scan.rows.wide, %scan.out.elements.wide
%gate.stride.0 = mul i32 %in.channels, %out.channels %scan.gate.stride.0.wide = mul i64 %scan.in.channels.wide, %scan.out.channels.wide %state.matrix = mul i32 %out.channels, %out.channels %scan.state.matrix.wide = mul i64 %scan.out.channels.wide, %scan.out.channels.wide
%gate.stride.1 = add i32 %gate.stride.0, %state.matrix %scan.gate.stride.1.wide = add i64 %scan.gate.stride.0.wide, %scan.state.matrix.wide
%delta.base = add i32 0, 0 %scan.delta.base.wide = add i64 0, 0 %gate2.batch = mul i32 %batch, 2 %scan.gate2.batch.wide = mul i64 %scan.batch.wide, 2
%row.gradient.base = mul i32 %gates, %batch %scan.row.gradient.base.wide = mul i64 %scan.gates.wide, %scan.batch.wide %rnn = icmp eq i32 %gates, 1
	%reverse.bias.span = select i1 %has.bias, i32 %out.channels, i32 0 %reverse.bias.span.wide = select i1 %has.bias, i64 %scan.out.channels.wide, i64 0
	%gate.stride = add i32 %gate.stride.1, %reverse.bias.span %scan.gate.stride.wide = add i64 %scan.gate.stride.1.wide, %reverse.bias.span.wide
%gru = icmp eq i32 %gates, 3 %lstm = icmp eq i32 %gates, 4 %simple = or i1 %rnn, %gru
%supported = or i1 %simple, %lstm br i1 %supported, label %row.loop, label %invalid row.loop:
%row = phi i32 [ %tid, %entry ], [ %row.next, %row.done ]
%row.more = icmp ult i32 %row, %rows br i1 %row.more, label %clear.gradient.loop, label %reduce.entry
clear.gradient.loop: %clear.p = phi i32 [ 0, %row.loop ], [ %clear.next, %clear.gradient.step ]
%scan.row.wide = zext i32 %row to i64 %scan.clear.p.wide = zext i32 %clear.p to i64 %row.gradient.offset = mul i32 %row, %parameters %scan.row.gradient.offset.wide = mul i64 %scan.row.wide, %scan.parameters.wide %scan.row.gradient.start.wide = add i64 %scan.row.gradient.base.wide, %scan.row.gradient.offset.wide %row.gradient.start = add i32 %row.gradient.base, %row.gradient.offset
%clear.more = icmp ult i32 %clear.p, %parameters br i1 %clear.more, label %clear.gradient.step, label %clear.state.loop
clear.gradient.step: %clear.index = add i32 %row.gradient.start, %clear.p %scan.clear.index.wide = add i64 %scan.row.gradient.start.wide, %scan.clear.p.wide
%clear.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.clear.index.wide
store float %state.zero, ptr addrspace(1) %clear.ptr, align 4 %clear.next = add nuw i32 %clear.p, 1
br label %clear.gradient.loop clear.state.loop:
%clear.h = phi i32 [ 0, %clear.gradient.loop ], [ %clear.h.next, %clear.state.step ]
%scratch.base.0 = mul i32 %rows, %parameters %scan.scratch.base.0.wide = mul i64 %scan.rows.wide, %scan.parameters.wide %scratch.base = add i32 %row.gradient.base, %scratch.base.0 %scan.scratch.base.wide = add i64 %scan.row.gradient.base.wide, %scan.scratch.base.0.wide
%scratch.row = mul i32 %row, %out.channels %scan.scratch.row.wide = mul i64 %scan.row.wide, %scan.out.channels.wide %dh.start = add i32 %scratch.base, %scratch.row %scan.dh.start.wide = add i64 %scan.scratch.base.wide, %scan.scratch.row.wide
%dc.base.0 = mul i32 %rows, %out.channels %scan.dc.base.0.wide = mul i64 %scan.rows.wide, %scan.out.channels.wide %dc.base = add i32 %scratch.base, %dc.base.0 %scan.dc.base.wide = add i64 %scan.scratch.base.wide, %scan.dc.base.0.wide
%dc.start = add i32 %dc.base, %scratch.row %scan.dc.start.wide = add i64 %scan.dc.base.wide, %scan.scratch.row.wide %clear.h.more = icmp ult i32 %clear.h, %out.channels
br i1 %clear.h.more, label %clear.state.step, label %time.loop clear.state.step:
%clear.dh.index = add i32 %dh.start, %clear.h %clear.dc.index = add i32 %dc.start, %clear.h %scan.clear.h.wide = zext i32 %clear.h to i64 %scan.clear.dh.index.wide = add i64 %scan.dh.start.wide, %scan.clear.h.wide %scan.clear.dc.index.wide = add i64 %scan.dc.start.wide, %scan.clear.h.wide
%clear.dh.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.clear.dh.index.wide
%clear.dc.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.clear.dc.index.wide
store float %state.zero, ptr addrspace(1) %clear.dh.ptr, align 4 store float %state.zero, ptr addrspace(1) %clear.dc.ptr, align 4
%clear.h.next = add nuw i32 %clear.h, 1 br label %clear.state.loop time.loop:
%time = phi i32 [ %length, %clear.state.loop ], [ %time.current, %time.done ] %time.current = sub i32 %time, 1 %scan.time.current.wide = zext i32 %time.current to i64 %scan.previous.time.wide = sub i64 %scan.time.current.wide, 1
%row.output.base = mul i32 %row, %out.elements %scan.row.output.base.wide = mul i64 %scan.row.wide, %scan.out.elements.wide %input.row.base = mul i32 %row, %in.elements %scan.input.row.base.wide = mul i64 %scan.row.wide, %scan.in.elements.wide
%previous.time = sub i32 %time.current, 1 %previous.exists = icmp sge i32 %previous.time, 0 %scan.previous.safe.wide = select i1 %previous.exists, i64 %scan.previous.time.wide, i64 0
%previous.safe = select i1 %previous.exists, i32 %previous.time, i32 0 %time.more = icmp sge i32 %time.current, 0
br i1 %time.more, label %scan.mode, label %row.done scan.mode:
br i1 %lstm, label %gate.delta.loop, label %rnn.check rnn.check:
br i1 %rnn, label %rnn.delta.loop, label %gru.delta.loop rnn.delta.loop:
%rnn.hidden = phi i32 [ 0, %rnn.check ], [ %rnn.next, %rnn.delta.step ]
%rnn.more = icmp ult i32 %rnn.hidden, %out.channels
br i1 %rnn.more, label %rnn.delta.step, label %delta.done rnn.delta.step:
%rnn.hidden.wide = zext i32 %rnn.hidden to i64 %rnn.hidden.base = mul i32 %rnn.hidden, %length %scan.rnn.hidden.base.wide = mul i64 %rnn.hidden.wide, %scan.length.wide %rnn.local = add i32 %rnn.hidden.base, %time.current %scan.rnn.local.wide = add i64 %scan.rnn.hidden.base.wide, %scan.time.current.wide
%rnn.index = add i32 %row.output.base, %rnn.local %scan.rnn.index.wide = add i64 %scan.row.output.base.wide, %scan.rnn.local.wide %rnn.dy.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %scan.rnn.index.wide %rnn.future.index = add i32 %dh.start, %rnn.hidden %scan.rnn.future.index.wide = add i64 %scan.dh.start.wide, %rnn.hidden.wide
%rnn.future.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.rnn.future.index.wide
%rnn.gate.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.rnn.index.wide
%rnn.dy = load float, ptr addrspace(1) %rnn.dy.ptr, align 4
%rnn.future = load float, ptr addrspace(1) %rnn.future.ptr, align 4
%rnn.gate.model = load i8, ptr addrspace(1) %rnn.gate.ptr, align 1 %rnn.gate = call float @recipe.decode(i8 %rnn.gate.model) %rnn.dh = call float @recipe.state.add(float %rnn.dy, float %rnn.future)
%rnn.square = call float @recipe.state.mul(float %rnn.gate, float %rnn.gate) %rnn.tanh.derivative = call float @recipe.state.sub(float %state.one, float %rnn.square)
%rnn.coded.relu = icmp eq i32 %cell.activation, 1 %rnn.coded.tanh = icmp eq i32 %cell.activation, 2 %rnn.coded.sigmoid = icmp eq i32 %cell.activation, 3
%rnn.positive = call i1 @recipe.ogt(i8 %rnn.gate.model, i8 0) %rnn.relu.derivative = select i1 %rnn.positive, float %state.one, float %state.zero
%rnn.one.minus = call float @recipe.state.sub(float %state.one, float %rnn.gate) %rnn.sigmoid.derivative = call float @recipe.state.mul(float %rnn.gate, float %rnn.one.minus)
%rnn.coded.a = select i1 %rnn.coded.relu, float %rnn.relu.derivative, float %state.one
%rnn.coded.b = select i1 %rnn.coded.tanh, float %rnn.tanh.derivative, float %rnn.coded.a
%rnn.coded.derivative = select i1 %rnn.coded.sigmoid, float %rnn.sigmoid.derivative, float %rnn.coded.b
%rnn.derivative = select i1 %coded, float %rnn.coded.derivative, float %rnn.tanh.derivative
%rnn.delta = call float @recipe.state.mul(float %rnn.dh, float %rnn.derivative) %rnn.delta.index = add i32 %delta.base, %rnn.index %scan.rnn.delta.index.wide = add i64 %scan.delta.base.wide, %scan.rnn.index.wide
%rnn.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.rnn.delta.index.wide
store float %rnn.delta, ptr addrspace(1) %rnn.delta.ptr, align 4 %rnn.next = add i32 %rnn.hidden, 1
br label %rnn.delta.loop gru.delta.loop: %gru.hidden = phi i32 [ 0, %rnn.check ], [ %gru.next, %gru.delta.step ]
%gru.more = icmp ult i32 %gru.hidden, %out.channels
br i1 %gru.more, label %gru.delta.step, label %gru.reset.loop gru.delta.step:
%gru.hidden.wide = zext i32 %gru.hidden to i64 %gru.hidden.base = mul i32 %gru.hidden, %length %scan.gru.hidden.base.wide = mul i64 %gru.hidden.wide, %scan.length.wide %gru.local = add i32 %gru.hidden.base, %time.current %scan.gru.local.wide = add i64 %scan.gru.hidden.base.wide, %scan.time.current.wide
%gru.index = add i32 %row.output.base, %gru.local %scan.gru.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.local.wide %gru.previous.local = add i32 %gru.hidden.base, %previous.safe %scan.gru.previous.local.wide = add i64 %scan.gru.hidden.base.wide, %scan.previous.safe.wide
%gru.previous.index = add i32 %row.output.base, %gru.previous.local %scan.gru.previous.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.previous.local.wide
%gru.dy.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %scan.gru.index.wide %gru.future.index = add i32 %dh.start, %gru.hidden %scan.gru.future.index.wide = add i64 %scan.dh.start.wide, %gru.hidden.wide
%gru.future.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.gru.future.index.wide
%gru.previous.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %scan.gru.previous.index.wide
%gru.z.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.gru.index.wide
%gru.n.index = add i32 %gru.index, %gate2.batch %scan.gru.n.index.wide = add i64 %scan.gru.index.wide, %scan.gate2.batch.wide
%gru.n.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.gru.n.index.wide
%gru.dy = load float, ptr addrspace(1) %gru.dy.ptr, align 4
%gru.future = load float, ptr addrspace(1) %gru.future.ptr, align 4
%gru.previous.loaded.model = load i8, ptr addrspace(1) %gru.previous.ptr, align 1 %gru.previous.loaded = call float @recipe.decode(i8 %gru.previous.loaded.model)
%gru.previous = select i1 %previous.exists, float %gru.previous.loaded, float %state.zero
%gru.z.model = load i8, ptr addrspace(1) %gru.z.ptr, align 1 %gru.z = call float @recipe.decode(i8 %gru.z.model)
%gru.n.model = load i8, ptr addrspace(1) %gru.n.ptr, align 1 %gru.n = call float @recipe.decode(i8 %gru.n.model) %gru.dh = call float @recipe.state.add(float %gru.dy, float %gru.future)
%gru.one.z = call float @recipe.state.sub(float %state.one, float %gru.z) %gru.z.difference = call float @recipe.state.sub(float %gru.previous, float %gru.n)
%gru.dz.0 = call float @recipe.state.mul(float %gru.dh, float %gru.z.difference) %gru.dz.1 = call float @recipe.state.mul(float %gru.dz.0, float %gru.z)
%gru.dz = call float @recipe.state.mul(float %gru.dz.1, float %gru.one.z) %gru.n.square = call float @recipe.state.mul(float %gru.n, float %gru.n)
%gru.n.derivative = call float @recipe.state.sub(float %state.one, float %gru.n.square) %gru.dn.0 = call float @recipe.state.mul(float %gru.dh, float %gru.one.z)
%gru.dn = call float @recipe.state.mul(float %gru.dn.0, float %gru.n.derivative) %gru.dz.index = add i32 %delta.base, %gru.index %scan.gru.dz.index.wide = add i64 %scan.delta.base.wide, %scan.gru.index.wide
%gru.dn.index.0 = add i32 %delta.base, %gate2.batch %scan.gru.dn.base.wide = add i64 %scan.delta.base.wide, %scan.gate2.batch.wide %scan.gru.dn.index.wide = add i64 %scan.gru.dn.base.wide, %scan.gru.index.wide
%gru.dz.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.gru.dz.index.wide
%gru.dn.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.gru.dn.index.wide
store float %gru.dz, ptr addrspace(1) %gru.dz.ptr, align 4
store float %gru.dn, ptr addrspace(1) %gru.dn.ptr, align 4 %gru.next = add i32 %gru.hidden, 1
br label %gru.delta.loop gru.reset.loop:
%gru.source = phi i32 [ 0, %gru.delta.loop ], [ %gru.source.next, %gru.reset.store ]
%gru.source.more = icmp ult i32 %gru.source, %out.channels %scan.gru.source.wide = zext i32 %gru.source to i64
br i1 %gru.source.more, label %gru.reset.sum.loop, label %delta.done gru.reset.sum.loop:
%gru.target = phi i32 [ 0, %gru.reset.loop ], [ %gru.target.next, %gru.reset.sum.step ]
%gru.reset.sum = phi float [ %state.zero, %gru.reset.loop ], [ %gru.reset.sum.next, %gru.reset.sum.step ]
%gru.target.more = icmp ult i32 %gru.target, %out.channels
br i1 %gru.target.more, label %gru.reset.sum.step, label %gru.reset.store gru.reset.sum.step:
%scan.gru.target.wide = zext i32 %gru.target to i64 %gru.candidate.base = mul i32 %gate.stride, 2 %scan.gru.candidate.base.wide = mul i64 %scan.gate.stride.wide, 2 %gru.candidate.state = add i32 %gru.candidate.base, %gate.stride.0 %scan.gru.candidate.state.wide = add i64 %scan.gru.candidate.base.wide, %scan.gate.stride.0.wide
%gru.weight.row = mul i32 %gru.source, %out.channels %scan.gru.weight.row.wide = mul i64 %scan.gru.source.wide, %scan.out.channels.wide %gru.weight.local = add i32 %gru.weight.row, %gru.target %scan.gru.weight.local.wide = add i64 %scan.gru.weight.row.wide, %scan.gru.target.wide
%gru.weight.index = add i32 %gru.candidate.state, %gru.weight.local %scan.gru.weight.index.wide = add i64 %scan.gru.candidate.state.wide, %scan.gru.weight.local.wide
%gru.weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %scan.gru.weight.index.wide
%gru.target.base = mul i32 %gru.target, %length %scan.gru.target.base.wide = mul i64 %scan.gru.target.wide, %scan.length.wide %gru.target.local = add i32 %gru.target.base, %time.current %scan.gru.target.local.wide = add i64 %scan.gru.target.base.wide, %scan.time.current.wide
%gru.target.index = add i32 %row.output.base, %gru.target.local %scan.gru.target.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.target.local.wide %gru.target.delta.0 = add i32 %delta.base, %gate2.batch %scan.gru.target.delta.0.wide = add i64 %scan.delta.base.wide, %scan.gate2.batch.wide
%gru.target.delta.index = add i32 %gru.target.delta.0, %gru.target.index %scan.gru.target.delta.index.wide = add i64 %scan.gru.target.delta.0.wide, %scan.gru.target.index.wide
%gru.target.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.gru.target.delta.index.wide
%gru.weight.model = load i8, ptr addrspace(1) %gru.weight.ptr, align 1 %gru.weight = call float @recipe.decode(i8 %gru.weight.model)
%gru.target.delta = load float, ptr addrspace(1) %gru.target.delta.ptr, align 4
%gru.reset.product = call float @recipe.state.mul(float %gru.weight, float %gru.target.delta)
%gru.reset.sum.next = call float @recipe.state.add(float %gru.reset.sum, float %gru.reset.product)
%gru.target.next = add i32 %gru.target, 1 br label %gru.reset.sum.loop gru.reset.store:
%gru.source.base = mul i32 %gru.source, %length %scan.gru.source.base.wide = mul i64 %scan.gru.source.wide, %scan.length.wide %gru.source.local = add i32 %gru.source.base, %time.current %scan.gru.source.local.wide = add i64 %scan.gru.source.base.wide, %scan.time.current.wide
%gru.source.index = add i32 %row.output.base, %gru.source.local %scan.gru.source.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.source.local.wide
%gru.source.previous.local = add i32 %gru.source.base, %previous.safe %scan.gru.source.previous.local.wide = add i64 %scan.gru.source.base.wide, %scan.previous.safe.wide
%gru.source.previous.index = add i32 %row.output.base, %gru.source.previous.local %scan.gru.source.previous.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.source.previous.local.wide
%gru.source.previous.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %scan.gru.source.previous.index.wide
%gru.r.index = add i32 %batch, %gru.source.index %scan.gru.r.index.wide = add i64 %scan.batch.wide, %scan.gru.source.index.wide
%gru.r.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.gru.r.index.wide
%gru.source.previous.loaded.model = load i8, ptr addrspace(1) %gru.source.previous.ptr, align 1 %gru.source.previous.loaded = call float @recipe.decode(i8 %gru.source.previous.loaded.model)
%gru.source.previous = select i1 %previous.exists, float %gru.source.previous.loaded, float %state.zero
%gru.r.model = load i8, ptr addrspace(1) %gru.r.ptr, align 1 %gru.r = call float @recipe.decode(i8 %gru.r.model)
%gru.dr = call float @recipe.state.mul(float %gru.reset.sum, float %gru.source.previous) %gru.one.r = call float @recipe.state.sub(float %state.one, float %gru.r)
%gru.dr.0 = call float @recipe.state.mul(float %gru.dr, float %gru.r) %gru.dr.1 = call float @recipe.state.mul(float %gru.dr.0, float %gru.one.r)
%gru.dr.base = add i32 %delta.base, %batch %scan.gru.dr.base.wide = add i64 %scan.delta.base.wide, %scan.batch.wide %gru.dr.index = add i32 %gru.dr.base, %gru.source.index %scan.gru.dr.index.wide = add i64 %scan.gru.dr.base.wide, %scan.gru.source.index.wide
%gru.dr.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.gru.dr.index.wide
store float %gru.dr.1, ptr addrspace(1) %gru.dr.ptr, align 4 %gru.source.next = add i32 %gru.source, 1
br label %gru.reset.loop gate.delta.loop: %hidden = phi i32 [ 0, %scan.mode ], [ %hidden.next, %gate.delta.step ]
%hidden.more = icmp ult i32 %hidden, %out.channels br i1 %hidden.more, label %gate.delta.step, label %delta.done
gate.delta.step: %hidden.wide = zext i32 %hidden to i64 %hidden.base = mul i32 %hidden, %length %scan.lstm.hidden.base.wide = mul i64 %hidden.wide, %scan.length.wide %local = add i32 %hidden.base, %time.current %scan.lstm.local.wide = add i64 %scan.lstm.hidden.base.wide, %scan.time.current.wide
%index = add i32 %row.output.base, %local %scan.lstm.index.wide = add i64 %scan.row.output.base.wide, %scan.lstm.local.wide %previous.local = add i32 %hidden.base, %previous.safe %scan.lstm.previous.local.wide = add i64 %scan.lstm.hidden.base.wide, %scan.previous.safe.wide
%previous.index = add i32 %row.output.base, %previous.local %scan.lstm.previous.index.wide = add i64 %scan.row.output.base.wide, %scan.lstm.previous.local.wide %cell.base = mul i32 %gates, %batch %scan.lstm.cell.base.wide = mul i64 %scan.gates.wide, %scan.batch.wide
%cell.index = add i32 %cell.base, %index %scan.lstm.cell.index.wide = add i64 %scan.lstm.cell.base.wide, %scan.lstm.index.wide %cell.previous.index = add i32 %cell.base, %previous.index %scan.lstm.cell.previous.index.wide = add i64 %scan.lstm.cell.base.wide, %scan.lstm.previous.index.wide
%dy.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %scan.lstm.index.wide %dh.index = add i32 %dh.start, %hidden %scan.lstm.dh.index.wide = add i64 %scan.dh.start.wide, %hidden.wide
%dc.index = add i32 %dc.start, %hidden %scan.lstm.dc.index.wide = add i64 %scan.dc.start.wide, %hidden.wide %dh.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.lstm.dh.index.wide
%dc.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.lstm.dc.index.wide
%cell.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.lstm.cell.index.wide
%cell.previous.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.lstm.cell.previous.index.wide
%dy = load float, ptr addrspace(1) %dy.ptr, align 4 %dh.future = load float, ptr addrspace(1) %dh.ptr, align 4
%dc.future = load float, ptr addrspace(1) %dc.ptr, align 4 %cell.model = load i8, ptr addrspace(1) %cell.ptr, align 1 %cell = call float @recipe.decode(i8 %cell.model)
%cell.previous.loaded.model = load i8, ptr addrspace(1) %cell.previous.ptr, align 1 %cell.previous.loaded = call float @recipe.decode(i8 %cell.previous.loaded.model)
%cell.previous = select i1 %previous.exists, float %cell.previous.loaded, float %state.zero
%i.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.lstm.index.wide %f.index = add i32 %batch, %index %scan.lstm.f.index.wide = add i64 %scan.batch.wide, %scan.lstm.index.wide
%o.index = add i32 %f.index, %batch %scan.lstm.o.index.wide = add i64 %scan.lstm.f.index.wide, %scan.batch.wide %g.index = add i32 %o.index, %batch %scan.lstm.g.index.wide = add i64 %scan.lstm.o.index.wide, %scan.batch.wide
%f.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.lstm.f.index.wide
%o.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.lstm.o.index.wide
%g.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.lstm.g.index.wide
%i.model = load i8, ptr addrspace(1) %i.ptr, align 1 %i = call float @recipe.decode(i8 %i.model) %f.model = load i8, ptr addrspace(1) %f.ptr, align 1 %f = call float @recipe.decode(i8 %f.model)
%o.model = load i8, ptr addrspace(1) %o.ptr, align 1 %o = call float @recipe.decode(i8 %o.model) %g.model = load i8, ptr addrspace(1) %g.ptr, align 1 %g = call float @recipe.decode(i8 %g.model)
%dh = call float @recipe.state.add(float %dy, float %dh.future) %cell.tanh = call float @recipe.state.tanh(float %cell)
%cell.tanh.square = call float @recipe.state.mul(float %cell.tanh, float %cell.tanh) %cell.tanh.derivative = call float @recipe.state.sub(float %state.one, float %cell.tanh.square)
%cell.chain.0 = call float @recipe.state.mul(float %dh, float %o) %cell.chain = call float @recipe.state.mul(float %cell.chain.0, float %cell.tanh.derivative)
%dc = call float @recipe.state.add(float %dc.future, float %cell.chain) %one.o = call float @recipe.state.sub(float %state.one, float %o) %do.0 = call float @recipe.state.mul(float %dh, float %cell.tanh)
%do.1 = call float @recipe.state.mul(float %do.0, float %o) %do = call float @recipe.state.mul(float %do.1, float %one.o) %one.i = call float @recipe.state.sub(float %state.one, float %i) %di.0 = call float @recipe.state.mul(float %dc, float %g)
%di.1 = call float @recipe.state.mul(float %di.0, float %i) %di = call float @recipe.state.mul(float %di.1, float %one.i) %one.f = call float @recipe.state.sub(float %state.one, float %f)
%df.0 = call float @recipe.state.mul(float %dc, float %cell.previous) %df.1 = call float @recipe.state.mul(float %df.0, float %f) %df = call float @recipe.state.mul(float %df.1, float %one.f)
%g.square = call float @recipe.state.mul(float %g, float %g) %one.g.square = call float @recipe.state.sub(float %state.one, float %g.square) %dg.0 = call float @recipe.state.mul(float %dc, float %i)
%dg = call float @recipe.state.mul(float %dg.0, float %one.g.square) %dc.previous = call float @recipe.state.mul(float %dc, float %f)
store float %dc.previous, ptr addrspace(1) %dc.ptr, align 4 %delta0.index = add i32 %delta.base, %index %scan.lstm.delta0.index.wide = add i64 %scan.delta.base.wide, %scan.lstm.index.wide
%delta1.index = add i32 %delta0.index, %batch %scan.lstm.delta1.index.wide = add i64 %scan.lstm.delta0.index.wide, %scan.batch.wide %delta2.index = add i32 %delta1.index, %batch %scan.lstm.delta2.index.wide = add i64 %scan.lstm.delta1.index.wide, %scan.batch.wide
%delta3.index = add i32 %delta2.index, %batch %scan.lstm.delta3.index.wide = add i64 %scan.lstm.delta2.index.wide, %scan.batch.wide
%delta0.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.lstm.delta0.index.wide
%delta1.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.lstm.delta1.index.wide
%delta2.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.lstm.delta2.index.wide
%delta3.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.lstm.delta3.index.wide
store float %di, ptr addrspace(1) %delta0.ptr, align 4 store float %df, ptr addrspace(1) %delta1.ptr, align 4
store float %do, ptr addrspace(1) %delta2.ptr, align 4 store float %dg, ptr addrspace(1) %delta3.ptr, align 4
%hidden.next = add nuw i32 %hidden, 1 br label %gate.delta.loop delta.done: br label %parameter.loop parameter.loop:
%p = phi i32 [ 0, %delta.done ], [ %p.next, %parameter.advance ] %p.more = icmp ult i32 %p, %parameters
br i1 %p.more, label %parameter.step, label %hidden.gradient.loop parameter.step:
%gate = udiv i32 %p, %gate.stride %within = urem i32 %p, %gate.stride
%input.weight = icmp ult i32 %within, %gate.stride.0
br i1 %input.weight, label %parameter.advance, label %parameter.value parameter.value:
%state.end = add i32 %gate.stride.0, %state.matrix %state.weight = icmp ult i32 %within, %state.end
%state.index = sub i32 %within, %gate.stride.0 %selected.index = select i1 %state.weight, i32 %state.index, i32 0
%source.channel = udiv i32 %selected.index, %out.channels %scan.source.channel.wide = zext i32 %source.channel to i64 %target.hidden = urem i32 %selected.index, %out.channels
%bias.hidden = sub i32 %within, %state.end %delta.hidden = select i1 %state.weight, i32 %target.hidden, i32 %bias.hidden
%delta.hidden.base = mul i32 %delta.hidden, %length %scan.delta.hidden.wide = zext i32 %delta.hidden to i64 %scan.delta.hidden.base.wide = mul i64 %scan.delta.hidden.wide, %scan.length.wide %delta.local = add i32 %delta.hidden.base, %time.current %scan.delta.local.wide = add i64 %scan.delta.hidden.base.wide, %scan.time.current.wide
%delta.row.local = add i32 %row.output.base, %delta.local %scan.delta.row.local.wide = add i64 %scan.row.output.base.wide, %scan.delta.local.wide %delta.gate.base = mul i32 %gate, %batch %scan.delta.gate.wide = zext i32 %gate to i64 %scan.delta.gate.base.wide = mul i64 %scan.delta.gate.wide, %scan.batch.wide
%delta.gate.local = add i32 %delta.base, %delta.gate.base %scan.delta.gate.local.wide = add i64 %scan.delta.base.wide, %scan.delta.gate.base.wide %delta.index = add i32 %delta.gate.local, %delta.row.local %scan.delta.index.wide = add i64 %scan.delta.gate.local.wide, %scan.delta.row.local.wide
%gate.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.delta.index.wide
%gate.delta = load float, ptr addrspace(1) %gate.delta.ptr, align 4
%state.hidden.base = mul i32 %source.channel, %length %scan.state.hidden.base.wide = mul i64 %scan.source.channel.wide, %scan.length.wide %state.local = add i32 %state.hidden.base, %previous.safe %scan.state.local.wide = add i64 %scan.state.hidden.base.wide, %scan.previous.safe.wide %state.index.value = add i32 %row.output.base, %state.local %scan.state.index.value.wide = add i64 %scan.row.output.base.wide, %scan.state.local.wide
%state.ptr = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %scan.state.index.value.wide
%state.loaded.model = load i8, ptr addrspace(1) %state.ptr, align 1 %state.loaded = call float @recipe.decode(i8 %state.loaded.model)
%state.value = select i1 %previous.exists, float %state.loaded, float %state.zero
%candidate.gate = icmp eq i32 %gate, 2 %gru.candidate = and i1 %gru, %candidate.gate
%parameter.reset.local = add i32 %state.hidden.base, %time.current %scan.parameter.reset.local.wide = add i64 %scan.state.hidden.base.wide, %scan.time.current.wide
%parameter.reset.row = add i32 %row.output.base, %parameter.reset.local %scan.parameter.reset.row.wide = add i64 %scan.row.output.base.wide, %scan.parameter.reset.local.wide
%parameter.reset.raw = add i32 %batch, %parameter.reset.row %scan.parameter.reset.raw.wide = add i64 %scan.batch.wide, %scan.parameter.reset.row.wide
%parameter.reset.index = select i1 %gru.candidate, i32 %parameter.reset.raw, i32 0 %scan.parameter.reset.index.wide = select i1 %gru.candidate, i64 %scan.parameter.reset.raw.wide, i64 0
%parameter.reset.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.parameter.reset.index.wide
%parameter.reset.model = load i8, ptr addrspace(1) %parameter.reset.ptr, align 1 %parameter.reset = call float @recipe.decode(i8 %parameter.reset.model)
%parameter.reset.state = call float @recipe.state.mul(float %parameter.reset, float %state.value)
%parameter.state = select i1 %gru.candidate, float %parameter.reset.state, float %state.value
%source.value = select i1 %state.weight, float %parameter.state, float %state.one
%contribution = call float @recipe.state.mul(float %source.value, float %gate.delta) %scan.p.wide = zext i32 %p to i64 %scan.row.gradient.index.wide = add i64 %scan.row.gradient.start.wide, %scan.p.wide %row.gradient.index = add i32 %row.gradient.start, %p
%row.gradient.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.row.gradient.index.wide
%row.gradient.old = load float, ptr addrspace(1) %row.gradient.ptr, align 4
%row.gradient.new = call float @recipe.state.add(float %row.gradient.old, float %contribution)
store float %row.gradient.new, ptr addrspace(1) %row.gradient.ptr, align 4
br label %parameter.advance parameter.advance:
%p.next = add nuw i32 %p, 1 br label %parameter.loop hidden.gradient.loop:
%state.channel = phi i32 [ 0, %parameter.loop ], [ %state.channel.next, %hidden.gradient.store ] %scan.state.channel.wide = zext i32 %state.channel to i64
%state.channel.more = icmp ult i32 %state.channel, %out.channels
br i1 %state.channel.more, label %hidden.gradient.sum.loop, label %time.done hidden.gradient.sum.loop:
%state.term = phi i32 [ 0, %hidden.gradient.loop ], [ %state.term.next, %hidden.gradient.sum.step ]
%state.sum = phi float [ %state.zero, %hidden.gradient.loop ], [ %state.sum.next, %hidden.gradient.sum.step ]
%state.terms = mul i32 %gates, %out.channels %state.term.more = icmp ult i32 %state.term, %state.terms
br i1 %state.term.more, label %hidden.gradient.sum.step, label %hidden.gradient.store hidden.gradient.sum.step:
%state.gate = udiv i32 %state.term, %out.channels %state.hidden = urem i32 %state.term, %out.channels %scan.state.gate.wide = zext i32 %state.gate to i64 %scan.state.hidden.wide = zext i32 %state.hidden to i64
%state.gate.base = mul i32 %state.gate, %gate.stride %scan.state.gate.base.wide = mul i64 %scan.state.gate.wide, %scan.gate.stride.wide %state.matrix.base = add i32 %state.gate.base, %gate.stride.0 %scan.state.matrix.base.wide = add i64 %scan.state.gate.base.wide, %scan.gate.stride.0.wide
%state.weight.row = mul i32 %state.channel, %out.channels %scan.state.weight.row.wide = mul i64 %scan.state.channel.wide, %scan.out.channels.wide %state.weight.local = add i32 %state.weight.row, %state.hidden %scan.state.weight.local.wide = add i64 %scan.state.weight.row.wide, %scan.state.hidden.wide
%state.weight.index = add i32 %state.matrix.base, %state.weight.local %scan.state.weight.index.wide = add i64 %scan.state.matrix.base.wide, %scan.state.weight.local.wide
%state.weight.ptr = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %scan.state.weight.index.wide
%state.delta.hidden.base = mul i32 %state.hidden, %length %scan.state.delta.hidden.base.wide = mul i64 %scan.state.hidden.wide, %scan.length.wide
%state.delta.local = add i32 %state.delta.hidden.base, %time.current %scan.state.delta.local.wide = add i64 %scan.state.delta.hidden.base.wide, %scan.time.current.wide
%state.delta.row = add i32 %row.output.base, %state.delta.local %scan.state.delta.row.wide = add i64 %scan.row.output.base.wide, %scan.state.delta.local.wide %state.delta.gate.base = mul i32 %state.gate, %batch %scan.state.delta.gate.base.wide = mul i64 %scan.state.gate.wide, %scan.batch.wide
%state.delta.base = add i32 %delta.base, %state.delta.gate.base %scan.state.delta.base.wide = add i64 %scan.delta.base.wide, %scan.state.delta.gate.base.wide
%state.delta.index = add i32 %state.delta.base, %state.delta.row %scan.state.delta.index.wide = add i64 %scan.state.delta.base.wide, %scan.state.delta.row.wide
%state.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.state.delta.index.wide
%state.weight.value.model = load i8, ptr addrspace(1) %state.weight.ptr, align 1 %state.weight.value = call float @recipe.decode(i8 %state.weight.value.model)
%state.delta.value = load float, ptr addrspace(1) %state.delta.ptr, align 4
%state.product = call float @recipe.state.mul(float %state.weight.value, float %state.delta.value) %state.candidate = icmp eq i32 %state.gate, 2
%state.gru.candidate = and i1 %gru, %state.candidate %state.reset.hidden.base = mul i32 %state.channel, %length %scan.state.reset.hidden.base.wide = mul i64 %scan.state.channel.wide, %scan.length.wide
%state.reset.local = add i32 %state.reset.hidden.base, %time.current %scan.state.reset.local.wide = add i64 %scan.state.reset.hidden.base.wide, %scan.time.current.wide
%state.reset.row = add i32 %row.output.base, %state.reset.local %scan.state.reset.row.wide = add i64 %scan.row.output.base.wide, %scan.state.reset.local.wide %state.reset.raw = add i32 %batch, %state.reset.row %scan.state.reset.raw.wide = add i64 %scan.batch.wide, %scan.state.reset.row.wide
%state.reset.index = select i1 %state.gru.candidate, i32 %state.reset.raw, i32 0 %scan.state.reset.index.wide = select i1 %state.gru.candidate, i64 %scan.state.reset.raw.wide, i64 0
%state.reset.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.state.reset.index.wide
%state.reset.model = load i8, ptr addrspace(1) %state.reset.ptr, align 1 %state.reset = call float @recipe.decode(i8 %state.reset.model)
%state.reset.product = call float @recipe.state.mul(float %state.product, float %state.reset)
%state.contribution = select i1 %state.gru.candidate, float %state.reset.product, float %state.product
%state.sum.next = call float @recipe.state.add(float %state.sum, float %state.contribution) %state.term.next = add nuw i32 %state.term, 1
br label %hidden.gradient.sum.loop hidden.gradient.store:
%state.dh.index = add i32 %dh.start, %state.channel %scan.state.dh.index.wide = add i64 %scan.dh.start.wide, %scan.state.channel.wide
%state.dh.ptr = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %scan.state.dh.index.wide
%state.direct.hidden.base = mul i32 %state.channel, %length %scan.state.direct.hidden.base.wide = mul i64 %scan.state.channel.wide, %scan.length.wide
%state.direct.local = add i32 %state.direct.hidden.base, %time.current %scan.state.direct.local.wide = add i64 %scan.state.direct.hidden.base.wide, %scan.time.current.wide
%state.direct.index = add i32 %row.output.base, %state.direct.local %scan.state.direct.index.wide = add i64 %scan.row.output.base.wide, %scan.state.direct.local.wide
%state.direct.delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i64 %scan.state.direct.index.wide
%state.direct.z.ptr = getelementptr inbounds i8, ptr addrspace(1) %context, i64 %scan.state.direct.index.wide
%state.direct.dy = load float, ptr addrspace(1) %state.direct.delta.ptr, align 4
%state.direct.future = load float, ptr addrspace(1) %state.dh.ptr, align 4
%state.direct.z.model = load i8, ptr addrspace(1) %state.direct.z.ptr, align 1 %state.direct.z = call float @recipe.decode(i8 %state.direct.z.model)
%state.direct.dh = call float @recipe.state.add(float %state.direct.dy, float %state.direct.future)
%state.direct.raw = call float @recipe.state.mul(float %state.direct.z, float %state.direct.dh)
%state.direct = select i1 %gru, float %state.direct.raw, float %state.zero
%state.total = call float @recipe.state.add(float %state.sum, float %state.direct)
store float %state.total, ptr addrspace(1) %state.dh.ptr, align 4 %state.channel.next = add nuw i32 %state.channel, 1
br label %hidden.gradient.loop time.done: br label %time.loop row.done: %row.next = add i32 %row, %threads
br label %row.loop reduce.entry: call void @grid_barrier(i32 %threads)
call void @reduce_rows_state(ptr addrspace(1) %backward, ptr addrspace(1) %gradient, i32 %rows, i32 %parameters, i32 %parameters, i32 %row.gradient.base, i32 %offset, i32 %threads)
br label %projection.entry
projection.entry: call void @grid_barrier(i32 %threads) br label %projection.loop projection.loop:
%projection.gate = phi i32 [ 0, %projection.entry ], [ %projection.next, %projection.step ]
%projection.more = icmp ult i32 %projection.gate, %gates
br i1 %projection.more, label %projection.step, label %exit projection.step:
%projection.gate.wide = zext i32 %projection.gate to i64 %projection.weight.offset = mul i32 %projection.gate, %gate.stride %projection.weight.offset.wide = mul i64 %projection.gate.wide, %scan.gate.stride.wide
%projection.weights = getelementptr inbounds i8, ptr addrspace(1) %weights, i64 %projection.weight.offset.wide
%projection.delta.gate = mul i32 %projection.gate, %batch %projection.delta.gate.wide = mul i64 %projection.gate.wide, %scan.batch.wide
%projection.delta.offset = add i32 %delta.base, %projection.delta.gate %projection.delta.offset.wide = add i64 %scan.delta.base.wide, %projection.delta.gate.wide
%projection.delta = getelementptr inbounds float, ptr addrspace(1) %backward, i64 %projection.delta.offset.wide
%projection.gradient.offset = add i32 %offset, %projection.weight.offset
call void @contraction_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %projection.weights, ptr addrspace(1) %output,
ptr addrspace(1) %projection.delta, ptr addrspace(1) %previous, ptr addrspace(1) %gradient, i1 %write.input, i1 false, i1 false, i1 false,
i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels, i32 %length, i32 0,
i32 %projection.gradient.offset, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k,
i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) call void @grid_barrier(i32 %threads) %projection.next = add i32 %projection.gate, 1 br label %projection.loop
invalid: call void @llvm.trap() br label %exit exit: ret void } attributes #0 = { nounwind } attributes #1 = { alwaysinline nounwind } attributes #3 = { noinline nounwind }
; Fully unroll the product loop so each insertelement uses a constant lane index
; and the accumulator vector remains in registers.
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
define internal i8 @recipe.kv.encode(i8 %value) #1 {
entry:
ret i8 %value
}
define internal i8 @recipe.kv.decode(i8 %value) #1 {
entry:
ret i8 %value
}
define internal float @recipe.kv.to.f32(i8 %value) #1 {
entry:
%state = call float @recipe.decode(i8 %value)
%result = fadd float %state, 0.0
ret float %result
}
define internal i8 @recipe.kv.from.f16(half %value) #1 {
entry:
%wide = fpext half %value to float
%state = fadd float %wide, 0.0
%result = call i8 @recipe.encode(float %state)
ret i8 %result
}
define internal float @recipe.kv.to.state(i8 %value) #1 {
entry:
%result = call float @recipe.decode(i8 %value)
ret float %result
}
define internal i8 @recipe.kv.from.state(float %value) #1 {
entry:
%result = call i8 @recipe.encode(float %value)
ret i8 %result
}

declare float @llvm.copysign.f32(float, float)
