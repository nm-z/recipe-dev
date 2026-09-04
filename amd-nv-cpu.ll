target triple = "amdgcn-amd-amdhsa"
; NUMERIC BEGIN
declare double @llvm.sqrt.f64(double) declare double @llvm.fabs.f64(double) declare double @llvm.floor.f64(double)
define internal double @recipe.add(double %left, double %right) #1 { entry: %result = fadd double %left, %right ret double %result }
define internal double @recipe.sub(double %left, double %right) #1 { entry: %result = fsub double %left, %right ret double %result }
define internal double @recipe.mul(double %left, double %right) #1 { entry: %result = fmul double %left, %right ret double %result }
define internal double @recipe.div(double %left, double %right) #1 { entry: %result = fdiv double %left, %right ret double %result }
define internal double @recipe.neg(double %value) #1 { entry: %result = fneg double %value ret double %result }
define internal i1 @recipe.oeq(double %left, double %right) #1 { entry: %result = fcmp oeq double %left, %right ret i1 %result }
define internal i1 @recipe.oge(double %left, double %right) #1 { entry: %result = fcmp oge double %left, %right ret i1 %result }
define internal i1 @recipe.ogt(double %left, double %right) #1 { entry: %result = fcmp ogt double %left, %right ret i1 %result }
define internal i1 @recipe.ole(double %left, double %right) #1 { entry: %result = fcmp ole double %left, %right ret i1 %result }
define internal i1 @recipe.olt(double %left, double %right) #1 { entry: %result = fcmp olt double %left, %right ret i1 %result }
define internal i1 @recipe.one(double %left, double %right) #1 { entry: %result = fcmp one double %left, %right ret i1 %result }
define internal i1 @recipe.ord(double %left, double %right) #1 { entry: %result = fcmp ord double %left, %right ret i1 %result }
define internal double @recipe.from.u1(i1 %value) #1 { entry: %result = uitofp i1 %value to double ret double %result }
define internal double @recipe.from.u32(i32 %value) #1 { entry: %result = uitofp i32 %value to double ret double %result }
define internal double @recipe.from.s32(i32 %value) #1 { entry: %result = sitofp i32 %value to double ret double %result }
define internal i32 @recipe.to.u32(double %value) #1 { entry: %result = fptoui double %value to i32 ret i32 %result }
define internal i32 @recipe.to.s32(double %value) #1 { entry: %result = fptosi double %value to i32 ret i32 %result }
define internal double @recipe.from.f32(float %value) #1 { entry: %result = fpext float %value to double ret double %result }
define internal double @recipe.from.f16(half %value) #1 { entry: %result = fpext half %value to double ret double %result }
define internal half @recipe.to.f16(double %value) #1 { entry: %result = fptrunc double %value to half ret half %result }
define internal double @recipe.abs(double %value) #1 { entry: %result = call double @llvm.fabs.f64(double %value) ret double %result }
define internal double @recipe.floor(double %value) #1 { entry: %result = call double @llvm.floor.f64(double %value) ret double %result }
define internal double @recipe.sqrt(double %value) #1 { entry: %result = call double @llvm.sqrt.f64(double %value) ret double %result }
; This whole region is a placeholder that the build replaces per precision. The
; transcendentals resolve to definitions the build emits, never to a backend
; library, so every device evaluates the same coefficients in its declared
; arithmetic type and in the same order.
define internal double @recipe.exp(double %value) #1 { entry: %result = call double @recipe.math.exp(double %value) ret double %result }
define internal double @recipe.tanh(double %value) #1 { entry: %result = call double @recipe.math.tanh(double %value) ret double %result }
define internal double @recipe.cos(double %value) #1 { entry: %result = call double @recipe.math.cos(double %value) ret double %result }
define internal double @recipe.sin(double %value) #1 { entry: %result = call double @recipe.math.sin(double %value) ret double %result }
define internal double @recipe.log(double %value) #1 { entry: %result = call double @recipe.math.log(double %value) ret double %result }
define internal void @recipe.set.format(i32 %exp, i32 %man) #1 { entry: ret void }
; NUMERIC END
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier() declare i64 @__ockl_steadyctr_u64()
declare void @llvm.trap() @contraction_tile = external addrspace(3) global [0 x double], align 16
define internal double @contraction_input(
ptr addrspace(1) %input, i32 %row.base, i32 %position, i32 %term, i32 %span, i32 %length, i1 %conv ) #1 { entry:
%channel = udiv i32 %term, %span %window = urem i32 %term, %span
%offset = select i1 %conv, i32 %window, i32 0 %channel.base = mul i32 %channel, %length
%local.0 = add i32 %channel.base, %position %local = add i32 %local.0, %offset
%index = add i32 %row.base, %local %ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %index
%value = load double, ptr addrspace(1) %ptr, align 8 ret double %value }
define internal double @contraction_delta(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %index
%delta.value = load double, ptr addrspace(1) %delta.ptr, align 8
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %index
%output.value = load double, ptr addrspace(1) %output.ptr, align 8
%positive = call i1 @recipe.ogt(double %output.value, double 0.0)
%activated = select i1 %positive, double %delta.value, double 0.0
br label %done
done:
%value = phi double [ %delta.value, %entry ], [ %activated, %activation ]
ret double %value
}
define internal <16 x double> @contraction_delta_vector16(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %index
%delta.value = load <16 x double>, ptr addrspace(1) %delta.ptr, align 8
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %index
%output.value = load <16 x double>, ptr addrspace(1) %output.ptr, align 8
br label %activation.loop
activation.loop:
%activation.index = phi i32 [ 0, %activation ], [ %activation.next, %activation.step ]
%activation.values = phi <16 x double> [ zeroinitializer, %activation ], [ %activation.values.next, %activation.step ]
%activation.more = icmp ult i32 %activation.index, 16
br i1 %activation.more, label %activation.step, label %done
activation.step:
%activation.output = extractelement <16 x double> %output.value, i32 %activation.index
%activation.delta = extractelement <16 x double> %delta.value, i32 %activation.index
%activation.positive = call i1 @recipe.ogt(double %activation.output, double 0.0)
%activation.value = select i1 %activation.positive, double %activation.delta, double 0.0
%activation.values.next = insertelement <16 x double> %activation.values, double %activation.value, i32 %activation.index
%activation.next = add i32 %activation.index, 1
br label %activation.loop
done:
%value = phi <16 x double> [ %delta.value, %entry ], [ %activation.values, %activation.loop ]
ret <16 x double> %value
}
define internal void @reduce_rows(ptr addrspace(1) %source, ptr addrspace(1) %target, i32 %rows, i32 %columns, i32 %stride, i32 %source.offset, i32 %target.offset, i32 %threads) #1 {
entry:
%reduce.lid = call i32 @recipe.local.id.x()
%reduce.group = call i32 @recipe.group.id.x()
%reduce.block = call i32 @recipe.workgroup.size.x()
%reduce.group.base = mul i32 %reduce.group, %reduce.block
%tid = add i32 %reduce.group.base, %reduce.lid
br label %parameter.loop
parameter.loop:
%parameter = phi i32 [ %tid, %entry ], [ %parameter.next, %store ]
%parameter.more = icmp ult i32 %parameter, %columns
br i1 %parameter.more, label %seed.load, label %exit
seed.load:
%target.index = add i32 %target.offset, %parameter
%target.ptr = getelementptr inbounds double, ptr addrspace(1) %target, i32 %target.index
%source.first.index = add i32 %source.offset, %parameter
%source.first.ptr = getelementptr inbounds double, ptr addrspace(1) %source, i32 %source.first.index
%source.first = load double, ptr addrspace(1) %source.first.ptr, align 8
br label %row.loop
row.loop:
%row = phi i32 [ 1, %seed.load ], [ %row.next, %row.step ]
%sum = phi double [ %source.first, %seed.load ], [ %sum.next, %row.step ]
%row.more = icmp ult i32 %row, %rows
br i1 %row.more, label %row.step, label %store
row.step:
%row.base = mul i32 %row, %stride
%source.local = add i32 %row.base, %parameter
%source.index = add i32 %source.offset, %source.local
%source.ptr = getelementptr inbounds double, ptr addrspace(1) %source, i32 %source.index
%source.value = load double, ptr addrspace(1) %source.ptr, align 8
%sum.next = call double @recipe.add(double %sum, double %source.value)
%row.next = add i32 %row, 1
br label %row.loop
store:
store double %sum, ptr addrspace(1) %target.ptr, align 8
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
define internal void @contraction_stage_column16(<16 x double> %values, i32 %base, i32 %r, i32 %column, i32 %stride) #1 {
entry:
br label %loop
loop:
%j = phi i32 [ 0, %entry ], [ %j.next, %step ]
%more = icmp ult i32 %j, 16
br i1 %more, label %step, label %done
step:
%value = extractelement <16 x double> %values, i32 %j
%row.local = add i32 %r, %j
%row = mul i32 %row.local, %stride
%local = add i32 %row, %column
%index = add i32 %base, %local
%ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double %value, ptr addrspace(3) %ptr, align 8
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
%a.index = call i32 @contraction_a_index(i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br label %store
b.step:
%b.p = sub i32 %p, %a.count
%b.k = udiv i32 %b.p, %b.missing
%b.local = urem i32 %b.p, %b.missing
%b.n = add i32 %n.count, %b.local
%b.base = mul i32 %tile.m, %tile.k
%b.local.index = call i32 @contraction_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k)
%b.index = add i32 %b.base, %b.local.index
br label %store
a.k.step:
%a.k.p = sub i32 %p, %output.count
%a.k.local = udiv i32 %a.k.p, %m.count
%a.k.value = add i32 %k.count, %a.k.local
%a.k.m = urem i32 %a.k.p, %m.count
%a.k.index = call i32 @contraction_a_index(i32 %a.k.value, i32 %a.k.m, i32 %tile.m, i32 %tile.k)
br label %store
b.k.step:
%b.k.p = sub i32 %p, %a.k.limit
%b.k.local = udiv i32 %b.k.p, %n.count
%b.k.value = add i32 %k.count, %b.k.local
%b.k.n = urem i32 %b.k.p, %n.count
%b.k.base = mul i32 %tile.m, %tile.k
%b.k.local.index = call i32 @contraction_b_index(i32 %b.k.value, i32 %b.k.n, i32 %tile.n, i32 %tile.k)
%b.k.index = add i32 %b.k.base, %b.k.local.index
br label %store
store:
%index = phi i32 [ %a.index, %a.step ], [ %b.index, %b.step ], [ %a.k.index, %a.k.step ], [ %b.k.index, %b.k.step ]
%ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double 0.0, ptr addrspace(3) %ptr, align 8
%next = add i32 %p, %block
br label %loop
exit:
ret void
}
define internal <RECIPE_REGISTER_M x double> @contraction_a_fragment(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k) #1 {
entry:
%index = call i32 @contraction_a_index(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_M x double>, ptr addrspace(3) %ptr, align 8
ret <RECIPE_REGISTER_M x double> %fragment
}
define internal <RECIPE_REGISTER_N x double> @contraction_b_fragment(i32 %k, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
%local = call i32 @contraction_b_index(i32 %k, i32 %output.n.base, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_N x double>, ptr addrspace(3) %ptr, align 8
ret <RECIPE_REGISTER_N x double> %fragment
}
define internal void @contraction_stage_a_fragment(<RECIPE_FRAGMENT_K x double> %fragment, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.k = add i32 %k, %element
%index = call i32 @contraction_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
%value = extractelement <RECIPE_FRAGMENT_K x double> %fragment, i32 %element
%target = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double %value, ptr addrspace(3) %target, align 8
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_a_columns(<RECIPE_FRAGMENT_K x double> %fragment, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.m = add i32 %m, %element
%index = call i32 @contraction_a_index(i32 %k, i32 %local.m, i32 %tile.m, i32 %tile.k)
%value = extractelement <RECIPE_FRAGMENT_K x double> %fragment, i32 %element
%target = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double %value, ptr addrspace(3) %target, align 8
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
; Stage a fragment of consecutive K for one channel. The B tile is k-major, so
; consecutive K lands one %tile.n row apart and the elements are placed through
; @contraction_b_index rather than stored as one contiguous vector.
define internal void @contraction_stage_b_terms(<RECIPE_FRAGMENT_K x double> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.k = add i32 %k, %element
%local = call i32 @contraction_b_index(i32 %local.k, i32 %n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <RECIPE_FRAGMENT_K x double> %fragment, i32 %element
%target = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double %value, ptr addrspace(3) %target, align 8
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_b_fragment(<RECIPE_FRAGMENT_K x double> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%local.n = add i32 %n, %element
%local = call i32 @contraction_b_index(i32 %k, i32 %local.n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <RECIPE_FRAGMENT_K x double> %fragment, i32 %element
%target = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double %value, ptr addrspace(3) %target, align 8
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_delta_a_fragment(<RECIPE_FRAGMENT_K x double> %delta, <RECIPE_FRAGMENT_K x double> %output, i1 %relu, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%delta.value = extractelement <RECIPE_FRAGMENT_K x double> %delta, i32 %element
%output.value = extractelement <RECIPE_FRAGMENT_K x double> %output, i32 %element
%positive = call i1 @recipe.ogt(double %output.value, double 0.0)
%active = select i1 %positive, double %delta.value, double 0.0
%value = select i1 %relu, double %active, double %delta.value
%local.k = add i32 %k, %element
%index = call i32 @contraction_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
%target = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double %value, ptr addrspace(3) %target, align 8
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
ptr addrspace(5) %sums, i1 %lane.active, i1 %lane.store, i32 %lid,
i32 %lane.k, i32 %k.lanes, i32 %output.lane, i32 %output.lanes,
i32 %output.m.base, i32 %output.n.base, i32 %m.count, i32 %n.count,
i32 %k.count, i32 %tile.m, i32 %tile.n, i32 %tile.k ) #1 {
entry:
%chunk.sums = alloca [RECIPE_CHUNK_VALUES x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5)
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%chunks.adjusted = add i32 %k.count, RECIPE_CHUNK_K
%chunks.numerator = sub i32 %chunks.adjusted, 1
%chunks = udiv i32 %chunks.numerator, RECIPE_CHUNK_K
%single.sum.slot = icmp eq i32 RECIPE_CHUNK_VALUES, RECIPE_REGISTER_COUNT
%publish.lane.width = mul i32 %output.lanes, RECIPE_REGISTER_COUNT
%chunk.first = select i1 %lane.active, i32 %lane.k, i32 %chunks
%local.lane = icmp eq i32 %k.lanes, 1
%few.chunks = icmp ule i32 %chunks, RECIPE_CONTRACTION_LOCAL_CHUNKS
%local = or i1 %local.lane, %few.chunks
%local.owner = icmp eq i32 %lane.k, 0
%local.owner.active = and i1 %lane.active, %local.owner
br i1 %local, label %local.owner.test, label %shared.chunk.loop
local.owner.test:
br i1 %local.owner.active, label %local.k.begin, label %exit
local.k.begin:
%local.sums.initial = load <RECIPE_REGISTER_COUNT x RECIPE_STATE>, ptr addrspace(5) %sums, align RECIPE_STATE_ALIGN
%local.a.initial = call <RECIPE_REGISTER_M x double> @contraction_a_fragment(i32 0, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.initial = call <RECIPE_REGISTER_N x double> @contraction_b_fragment(i32 0, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %local.k.loop
local.k.loop:
%local.k = phi i32 [ 0, %local.k.begin ], [ %local.k.next, %local.product.done ]
%local.sums = phi <RECIPE_REGISTER_COUNT x RECIPE_STATE> [ %local.sums.initial, %local.k.begin ], [ %local.sums.current, %local.product.done ]
%local.a.fragment = phi <RECIPE_REGISTER_M x double> [ %local.a.initial, %local.k.begin ], [ %local.a.next, %local.product.done ]
%local.b.fragment = phi <RECIPE_REGISTER_N x double> [ %local.b.initial, %local.k.begin ], [ %local.b.next, %local.product.done ]
%local.k.next = add i32 %local.k, 1
%local.k.more = icmp ult i32 %local.k.next, %k.count
%local.k.prefetch = select i1 %local.k.more, i32 %local.k.next, i32 %local.k
%local.a.next = call <RECIPE_REGISTER_M x double> @contraction_a_fragment(i32 %local.k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.next = call <RECIPE_REGISTER_N x double> @contraction_b_fragment(i32 %local.k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%local.a.wide = call <RECIPE_REGISTER_M x RECIPE_STATE> @contraction_widen_m(<RECIPE_REGISTER_M x double> %local.a.fragment)
br label %local.product.loop
local.product.loop:
%local.product = phi i32 [ 0, %local.k.loop ], [ %local.product.next, %local.product.step ]
%local.sums.current = phi <RECIPE_REGISTER_COUNT x RECIPE_STATE> [ %local.sums, %local.k.loop ], [ %local.candidate, %local.product.step ]
%local.product.more = icmp ult i32 %local.product, RECIPE_REGISTER_COUNT
br i1 %local.product.more, label %local.product.step, label %local.product.done
local.product.step:
%local.a.index = urem i32 %local.product, RECIPE_REGISTER_M
%local.b.index = udiv i32 %local.product, RECIPE_REGISTER_M
%local.a = extractelement <RECIPE_REGISTER_M x RECIPE_STATE> %local.a.wide, i32 %local.a.index
%local.b = extractelement <RECIPE_REGISTER_N x double> %local.b.fragment, i32 %local.b.index
%local.b.wide = call RECIPE_STATE @recipe.decode(double %local.b)
%local.sum = extractelement <RECIPE_REGISTER_COUNT x RECIPE_STATE> %local.sums.current, i32 %local.product
%local.value = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %local.sum, RECIPE_STATE %local.a, RECIPE_STATE %local.b.wide)
%local.candidate = insertelement <RECIPE_REGISTER_COUNT x RECIPE_STATE> %local.sums.current, RECIPE_STATE %local.value, i32 %local.product
%local.product.next = add i32 %local.product, 1
br label %local.product.loop, !llvm.loop !0
local.product.done:
br i1 %local.k.more, label %local.k.loop, label %local.store
local.store:
store <RECIPE_REGISTER_COUNT x RECIPE_STATE> %local.sums.current, ptr addrspace(5) %sums, align RECIPE_STATE_ALIGN
br label %exit
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
%zero.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %chunk.sums, i32 %zero.index
store RECIPE_STATE %state.zero, ptr addrspace(5) %zero.ptr, align RECIPE_STATE_ALIGN
%zero.next = add i32 %zero.r, 1
br label %chunk.zero.loop
k.begin:
%k.first = mul i32 %chunk, RECIPE_CHUNK_K
%k.limit.raw = add i32 %k.first, RECIPE_CHUNK_K
%k.over = icmp ugt i32 %k.limit.raw, %k.count
%k.limit = select i1 %k.over, i32 %k.count, i32 %k.limit.raw
%slot.sum.base = mul i32 %sum.slot, RECIPE_REGISTER_COUNT
%a.initial = call <RECIPE_REGISTER_M x double> @contraction_a_fragment(i32 %k.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.initial = call <RECIPE_REGISTER_N x double> @contraction_b_fragment(i32 %k.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %k.loop
k.loop:
%k = phi i32 [ %k.first, %k.begin ], [ %k.next, %register.done ]
%a.fragment = phi <RECIPE_REGISTER_M x double> [ %a.initial, %k.begin ], [ %a.next, %register.done ]
%b.fragment = phi <RECIPE_REGISTER_N x double> [ %b.initial, %k.begin ], [ %b.next, %register.done ]
%k.next = add i32 %k, 1
%k.more = icmp ult i32 %k.next, %k.limit
%k.prefetch = select i1 %k.more, i32 %k.next, i32 %k
%a.next = call <RECIPE_REGISTER_M x double> @contraction_a_fragment(i32 %k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.next = call <RECIPE_REGISTER_N x double> @contraction_b_fragment(i32 %k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%a.wide = call <RECIPE_REGISTER_M x RECIPE_STATE> @contraction_widen_m(<RECIPE_REGISTER_M x double> %a.fragment)
br label %register.loop
register.loop:
%register.n = phi i32 [ 0, %k.loop ], [ %register.n.next, %register.next ]
%register.more = icmp ult i32 %register.n, RECIPE_REGISTER_N
br i1 %register.more, label %register.step, label %register.done
register.step:
%output.n.raw = add i32 %output.n.base, %register.n
%output.n.valid = icmp ult i32 %output.n.raw, %n.count
%b = extractelement <RECIPE_REGISTER_N x double> %b.fragment, i32 %register.n
%b.wide = call RECIPE_STATE @recipe.decode(double %b)
%b.seed = insertelement <RECIPE_REGISTER_M x RECIPE_STATE> poison, RECIPE_STATE %b.wide, i32 0
%b.vector = shufflevector <RECIPE_REGISTER_M x RECIPE_STATE> %b.seed, <RECIPE_REGISTER_M x RECIPE_STATE> poison, <RECIPE_REGISTER_M x i32> zeroinitializer
%register.base = mul i32 %register.n, RECIPE_REGISTER_M
%sum.index = add i32 %slot.sum.base, %register.base
%sum.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %chunk.sums, i32 %sum.index
%sum = load <RECIPE_REGISTER_M x RECIPE_STATE>, ptr addrspace(5) %sum.ptr, align RECIPE_STATE_ALIGN
%candidate = call <RECIPE_REGISTER_M x RECIPE_STATE> @recipe.state.madd.vector(<RECIPE_REGISTER_M x RECIPE_STATE> %sum, <RECIPE_REGISTER_M x RECIPE_STATE> %a.wide, <RECIPE_REGISTER_M x RECIPE_STATE> %b.vector)
store <RECIPE_REGISTER_M x RECIPE_STATE> %candidate, ptr addrspace(5) %sum.ptr, align RECIPE_STATE_ALIGN
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
call void @recipe.local.barrier()
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
%publish.source = getelementptr RECIPE_STATE, ptr addrspace(5) %chunk.sums, i32 %publish.source.index
%publish.value = load RECIPE_STATE, ptr addrspace(5) %publish.source, align RECIPE_STATE_ALIGN
%publish.row = mul i32 %publish.chunk, %output.lanes
%publish.column = add i32 %publish.row, %output.lane
%publish.target.base = mul i32 %publish.column, RECIPE_REGISTER_COUNT
%publish.target.index = add i32 %publish.target.base, %publish.r
%publish.target = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %publish.target.index
store RECIPE_STATE %publish.value, ptr addrspace(3) %publish.target, align RECIPE_STATE_ALIGN
%publish.r.next = add i32 %publish.r, 1
br label %publish.sum.loop
publish.finish:
%publish.chunk.next = add i32 %publish.chunk, %k.lanes
%publish.slot.next = add i32 %publish.slot, 1
br label %publish.loop
publish.done:
call void @recipe.local.barrier()
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
%fold.source = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %fold.source.index
%fold.value = load RECIPE_STATE, ptr addrspace(3) %fold.source, align RECIPE_STATE_ALIGN
%fold.target = getelementptr RECIPE_STATE, ptr addrspace(5) %sums, i32 %fold.r
%fold.current = load RECIPE_STATE, ptr addrspace(5) %fold.target, align RECIPE_STATE_ALIGN
%fold.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %fold.current, RECIPE_STATE %fold.value)
store RECIPE_STATE %fold.next, ptr addrspace(5) %fold.target, align RECIPE_STATE_ALIGN
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
ptr addrspace(5) %sums, ptr addrspace(1) %destination,
i1 %enable, i1 %first, i1 %last, i32 %lid, i32 %block,
i32 %n.base, i32 %n.count, i32 %r.count, i32 %out.channels, i32 %window,
i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %store.offset ) #1 { entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) br i1 %enable, label %channel.loop, label %exit
channel.loop:
%channel = phi i32 [ %lid, %entry ], [ %channel.next, %channel.done ] %slot = phi i32 [ 0, %entry ], [ %slot.next, %channel.done ]
%channel.more = icmp ult i32 %channel, %n.count br i1 %channel.more, label %channel.begin, label %exit
channel.begin:
%sum.ptr = getelementptr [RECIPE_REGISTER_N x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %slot
%previous = load RECIPE_STATE, ptr addrspace(5) %sum.ptr, align RECIPE_STATE_ALIGN %initial = select i1 %first, RECIPE_STATE %zero, RECIPE_STATE %previous br label %r.loop
r.loop:
%r = phi i32 [ 0, %channel.begin ], [ %r.next, %r.step ] %sum = phi RECIPE_STATE [ %initial, %channel.begin ], [ %sum.next, %r.step ]
%r.more = icmp ult i32 %r, %r.count br i1 %r.more, label %r.step, label %sum.store
r.step:
%base = mul i32 %tile.m, %tile.k %local = call i32 @contraction_b_index(i32 %r, i32 %channel, i32 %tile.n, i32 %tile.k) %index = add i32 %base, %local
%ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
%raw = load double, ptr addrspace(3) %ptr, align 8 %value = call RECIPE_STATE @recipe.decode(double %raw) %sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %value)
%r.next = add i32 %r, 1 br label %r.loop
sum.store:
store RECIPE_STATE %sum, ptr addrspace(5) %sum.ptr, align RECIPE_STATE_ALIGN br i1 %last, label %destination.store, label %channel.done
destination.store:
%filter = add i32 %n.base, %channel %bias.base = mul i32 %out.channels, %window %bias.local = add i32 %bias.base, %filter %bias.index = add i32 %store.offset, %bias.local
%bias.ptr = getelementptr inbounds double, ptr addrspace(1) %destination, i32 %bias.index
%bias = call double @recipe.encode(RECIPE_STATE %sum) store double %bias, ptr addrspace(1) %bias.ptr, align 8 br label %channel.done
channel.done:
%channel.next = add i32 %channel, %block %slot.next = add i32 %slot, 1 br label %channel.loop
exit: ret void
}
; Widen a staged A fragment to the arithmetic type once per K step, so the inner
; register loop never converts.
define internal <RECIPE_REGISTER_M x RECIPE_STATE> @contraction_widen_m(<RECIPE_REGISTER_M x double> %source) #1 {
entry:
br label %loop
loop:
%p = phi i32 [ 0, %entry ], [ %p.next, %step ]
%result = phi <RECIPE_REGISTER_M x RECIPE_STATE> [ poison, %entry ], [ %next, %step ]
%more = icmp ult i32 %p, RECIPE_REGISTER_M
br i1 %more, label %step, label %done
step:
%value = extractelement <RECIPE_REGISTER_M x double> %source, i32 %p
%wide = call RECIPE_STATE @recipe.decode(double %value)
%next = insertelement <RECIPE_REGISTER_M x RECIPE_STATE> %result, RECIPE_STATE %wide, i32 %p
%p.next = add i32 %p, 1
br label %loop
done:
ret <RECIPE_REGISTER_M x RECIPE_STATE> %result
}
; RECIPE_WMMA gfx11-f16 call @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16( || gfx11-bf16 call @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16( || gfx11-int8 definition declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 immarg, <4 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i1 immarg)\ndefine internal <8 x float> @recipe.wmma(<16 x i8> %a, <16 x i8> %b, <8 x float> %state) #1 { entry: %a.packed = bitcast <16 x i8> %a to <4 x i32> %b.packed = bitcast <16 x i8> %b to <4 x i32> %product = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 true, <4 x i32> %a.packed, i1 true, <4 x i32> %b.packed, <8 x i32> zeroinitializer, i1 false) %wide = sitofp <8 x i32> %product to <8 x float> %result = fadd <8 x float> %state, %wide ret <8 x float> %result }\n || gfx11-int4 definition declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 immarg, <2 x i32>, i1 immarg, <2 x i32>, <8 x i32>, i1 immarg)\ndefine internal i32 @recipe.pack.i4.word(i32 %bytes) #1 { entry: %nibbles = and i32 %bytes, 252645135 %pair.shift = lshr i32 %nibbles, 4 %pair.raw = or i32 %nibbles, %pair.shift %pair = and i32 %pair.raw, 16711935 %word.shift = lshr i32 %pair, 8 %word.raw = or i32 %pair, %word.shift %word = and i32 %word.raw, 65535 ret i32 %word }\ndefine internal <2 x i32> @recipe.pack.i4(<16 x i8> %values) #1 { entry: %bytes = bitcast <16 x i8> %values to <4 x i32> %bytes.0 = extractelement <4 x i32> %bytes, i32 0 %bytes.1 = extractelement <4 x i32> %bytes, i32 1 %bytes.2 = extractelement <4 x i32> %bytes, i32 2 %bytes.3 = extractelement <4 x i32> %bytes, i32 3 %word.0 = call i32 @recipe.pack.i4.word(i32 %bytes.0) %word.1 = call i32 @recipe.pack.i4.word(i32 %bytes.1) %word.2 = call i32 @recipe.pack.i4.word(i32 %bytes.2) %word.3 = call i32 @recipe.pack.i4.word(i32 %bytes.3) %word.1.high = shl i32 %word.1, 16 %packed.0 = or i32 %word.0, %word.1.high %word.3.high = shl i32 %word.3, 16 %packed.1 = or i32 %word.2, %word.3.high %result.0 = insertelement <2 x i32> poison, i32 %packed.0, i32 0 %result = insertelement <2 x i32> %result.0, i32 %packed.1, i32 1 ret <2 x i32> %result }\ndefine internal <8 x float> @recipe.wmma(<16 x i8> %a, <16 x i8> %b, <8 x float> %state) #1 { entry: %a.packed = call <2 x i32> @recipe.pack.i4(<16 x i8> %a) %b.packed = call <2 x i32> @recipe.pack.i4(<16 x i8> %b) %product = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 true, <2 x i32> %a.packed, i1 true, <2 x i32> %b.packed, <8 x i32> zeroinitializer, i1 false) %wide = sitofp <8 x i32> %product to <8 x float> %result = fadd <8 x float> %state, %wide ret <8 x float> %result }\n || gfx12-f16 definition declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)\ndefine internal <8 x float> @recipe.wmma(<16 x half> %a, <16 x half> %b, <8 x float> %state) #1 { entry: %a.low = shufflevector <16 x half> %a, <16 x half> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> %a.high = shufflevector <16 x half> %a, <16 x half> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> %b.low = shufflevector <16 x half> %b, <16 x half> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> %b.high = shufflevector <16 x half> %b, <16 x half> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> %first = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %a.low, <8 x half> %b.low, <8 x float> %state) %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %a.high, <8 x half> %b.high, <8 x float> %first) ret <8 x float> %result }\n || gfx12-bf16 definition declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16(<8 x i16>, <8 x i16>, <8 x float>)\ndefine internal <8 x float> @recipe.wmma(<16 x i16> %a, <16 x i16> %b, <8 x float> %state) #1 { entry: %a.low = shufflevector <16 x i16> %a, <16 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> %a.high = shufflevector <16 x i16> %a, <16 x i16> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> %b.low = shufflevector <16 x i16> %b, <16 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> %b.high = shufflevector <16 x i16> %b, <16 x i16> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> %first = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16(<8 x i16> %a.low, <8 x i16> %b.low, <8 x float> %state) %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16(<8 x i16> %a.high, <8 x i16> %b.high, <8 x float> %first) ret <8 x float> %result }\n || gfx12-int8 definition declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32(i1 immarg, <2 x i32>, i1 immarg, <2 x i32>, <8 x i32>, i1 immarg)\ndefine internal <8 x float> @recipe.wmma(<16 x i8> %a, <16 x i8> %b, <8 x float> %state) #1 { entry: %a.low.values = shufflevector <16 x i8> %a, <16 x i8> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> %a.high.values = shufflevector <16 x i8> %a, <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> %b.low.values = shufflevector <16 x i8> %b, <16 x i8> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> %b.high.values = shufflevector <16 x i8> %b, <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> %a.low = bitcast <8 x i8> %a.low.values to <2 x i32> %a.high = bitcast <8 x i8> %a.high.values to <2 x i32> %b.low = bitcast <8 x i8> %b.low.values to <2 x i32> %b.high = bitcast <8 x i8> %b.high.values to <2 x i32> %first = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32(i1 true, <2 x i32> %a.low, i1 true, <2 x i32> %b.low, <8 x i32> zeroinitializer, i1 false) %product = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32(i1 true, <2 x i32> %a.high, i1 true, <2 x i32> %b.high, <8 x i32> %first, i1 false) %wide = sitofp <8 x i32> %product to <8 x float> %result = fadd <8 x float> %state, %wide ret <8 x float> %result }\n


declare <8 x RECIPE_STATE> @recipe.wmma(<16 x double>, <16 x double>, <8 x RECIPE_STATE>)
; Matrix arithmetic consumes the operands staged by the common contraction
; composer, writes one state-width partial tile beside them, then maps that tile
; back into the composer's register ownership. Scheduling, tails, K tiling, and
; the epilogue therefore remain identical to the vector method.
define internal void @contraction_matrix_accumulate(
ptr addrspace(5) %sums, i1 %lane.active, i1 %lane.store, i32 %lid,
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
%matrix.first = phi <8 x RECIPE_STATE> [ zeroinitializer, %entry ], [ %matrix.first.next, %matrix.k.done ]
%matrix.second = phi <8 x RECIPE_STATE> [ zeroinitializer, %entry ], [ %matrix.second.next, %matrix.k.done ]
%matrix.third = phi <8 x RECIPE_STATE> [ zeroinitializer, %entry ], [ %matrix.third.next, %matrix.k.done ]
%matrix.fourth = phi <8 x RECIPE_STATE> [ zeroinitializer, %entry ], [ %matrix.fourth.next, %matrix.k.done ]
%matrix.k.more = icmp ult i32 %matrix.k, %matrix.k.rounded
br i1 %matrix.k.more, label %matrix.full, label %matrix.store.loop
matrix.full:
%matrix.a.index = call i32 @contraction_a_index(i32 %matrix.k, i32 %m.safe, i32 %tile.m, i32 %tile.k)
%matrix.a.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.a.index
%matrix.a.loaded = load <16 x double>, ptr addrspace(3) %matrix.a.ptr, align 16
%matrix.a.full = select i1 %m.valid, <16 x double> %matrix.a.loaded, <16 x double> zeroinitializer
%matrix.b.base = mul i32 %tile.m, %tile.k
%matrix.b.first.local = call i32 @contraction_b_index(i32 %matrix.k, i32 %n.first.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.first.index = add i32 %matrix.b.base, %matrix.b.first.local
%matrix.b.first.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.b.first.index
%matrix.b.first.loaded = load <16 x double>, ptr addrspace(3) %matrix.b.first.ptr, align 16
%matrix.b.first.full = select i1 %n.first.valid, <16 x double> %matrix.b.first.loaded, <16 x double> zeroinitializer
%matrix.b.second.local = call i32 @contraction_b_index(i32 %matrix.k, i32 %n.second.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.second.index = add i32 %matrix.b.base, %matrix.b.second.local
%matrix.b.second.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.b.second.index
%matrix.b.second.loaded = load <16 x double>, ptr addrspace(3) %matrix.b.second.ptr, align 16
%matrix.b.second.full = select i1 %n.second.valid, <16 x double> %matrix.b.second.loaded, <16 x double> zeroinitializer
%matrix.b.third.local = call i32 @contraction_b_index(i32 %matrix.k, i32 %n.third.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.third.index = add i32 %matrix.b.base, %matrix.b.third.local
%matrix.b.third.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.b.third.index
%matrix.b.third.loaded = load <16 x double>, ptr addrspace(3) %matrix.b.third.ptr, align 16
%matrix.b.third.full = select i1 %n.third.valid, <16 x double> %matrix.b.third.loaded, <16 x double> zeroinitializer
%matrix.b.fourth.local = call i32 @contraction_b_index(i32 %matrix.k, i32 %n.fourth.safe, i32 %tile.n, i32 %tile.k)
%matrix.b.fourth.index = add i32 %matrix.b.base, %matrix.b.fourth.local
%matrix.b.fourth.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.b.fourth.index
%matrix.b.fourth.loaded = load <16 x double>, ptr addrspace(3) %matrix.b.fourth.ptr, align 16
%matrix.b.fourth.full = select i1 %n.fourth.valid, <16 x double> %matrix.b.fourth.loaded, <16 x double> zeroinitializer
%matrix.first.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %matrix.a.full, <16 x double> %matrix.b.first.full, <8 x RECIPE_STATE> %matrix.first)
%matrix.second.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %matrix.a.full, <16 x double> %matrix.b.second.full, <8 x RECIPE_STATE> %matrix.second)
%matrix.third.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %matrix.a.full, <16 x double> %matrix.b.third.full, <8 x RECIPE_STATE> %matrix.third)
%matrix.fourth.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %matrix.a.full, <16 x double> %matrix.b.fourth.full, <8 x RECIPE_STATE> %matrix.fourth)
br label %matrix.k.done
matrix.k.done:
%matrix.k.next = add i32 %matrix.k, 16
br label %matrix.k.loop
matrix.store.loop:
%matrix.register = phi i32 [ 0, %matrix.k.loop ], [ %matrix.register.next, %matrix.store.step ]
%matrix.register.more = icmp ult i32 %matrix.register, 8
br i1 %matrix.register.more, label %matrix.store.step, label %matrix.exit
matrix.store.step:
%matrix.first.value = extractelement <8 x RECIPE_STATE> %matrix.first, i32 %matrix.register
%matrix.first.target = getelementptr RECIPE_STATE, ptr addrspace(5) %sums, i32 %matrix.register
%matrix.first.current = load RECIPE_STATE, ptr addrspace(5) %matrix.first.target, align RECIPE_STATE_ALIGN
%matrix.first.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %matrix.first.current, RECIPE_STATE %matrix.first.value)
store RECIPE_STATE %matrix.first.sum, ptr addrspace(5) %matrix.first.target, align RECIPE_STATE_ALIGN
%matrix.second.register = add i32 %matrix.register, 8
%matrix.second.value = extractelement <8 x RECIPE_STATE> %matrix.second, i32 %matrix.register
%matrix.second.target = getelementptr RECIPE_STATE, ptr addrspace(5) %sums, i32 %matrix.second.register
%matrix.second.current = load RECIPE_STATE, ptr addrspace(5) %matrix.second.target, align RECIPE_STATE_ALIGN
%matrix.second.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %matrix.second.current, RECIPE_STATE %matrix.second.value)
store RECIPE_STATE %matrix.second.sum, ptr addrspace(5) %matrix.second.target, align RECIPE_STATE_ALIGN
%matrix.third.register = add i32 %matrix.register, 16
%matrix.third.value = extractelement <8 x RECIPE_STATE> %matrix.third, i32 %matrix.register
%matrix.third.target = getelementptr RECIPE_STATE, ptr addrspace(5) %sums, i32 %matrix.third.register
%matrix.third.current = load RECIPE_STATE, ptr addrspace(5) %matrix.third.target, align RECIPE_STATE_ALIGN
%matrix.third.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %matrix.third.current, RECIPE_STATE %matrix.third.value)
store RECIPE_STATE %matrix.third.sum, ptr addrspace(5) %matrix.third.target, align RECIPE_STATE_ALIGN
%matrix.fourth.register = add i32 %matrix.register, 24
%matrix.fourth.value = extractelement <8 x RECIPE_STATE> %matrix.fourth, i32 %matrix.register
%matrix.fourth.target = getelementptr RECIPE_STATE, ptr addrspace(5) %sums, i32 %matrix.fourth.register
%matrix.fourth.current = load RECIPE_STATE, ptr addrspace(5) %matrix.fourth.target, align RECIPE_STATE_ALIGN
%matrix.fourth.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %matrix.fourth.current, RECIPE_STATE %matrix.fourth.value)
store RECIPE_STATE %matrix.fourth.sum, ptr addrspace(5) %matrix.fourth.target, align RECIPE_STATE_ALIGN
%matrix.register.next = add i32 %matrix.register, 1
br label %matrix.store.loop
matrix.exit:
ret void
}
define internal void @contraction_forward_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %kernel,
i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads ) #1 { entry:
; The running sums live in the arithmetic type for the whole K extent and are
; rounded to the model type once, at the store. Staging the operands in tiles
; therefore cannot move a rounding point.
%sums = alloca [RECIPE_REGISTER_COUNT x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5) %state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %lid = call i32 @recipe.local.id.x() %group = call i32 @recipe.group.id.x() %block = call i32 @recipe.workgroup.size.x() %groups = udiv i32 %threads, %block
%in.elements = mul i32 %in.channels, %in.length %out.elements = mul i32 %out.channels, %out.length %is.conv = icmp ne i32 %kernel, 0 %span = select i1 %is.conv, i32 %kernel, i32 1 %terms = mul i32 %in.channels, %span %m.total = mul i32 %rows, %out.length
%m.short = icmp ult i32 %tile.m, %m.total %m.tile = select i1 %m.short, i32 %tile.m, i32 %m.total %n.short = icmp ult i32 %tile.n, %out.channels %n.tile = select i1 %n.short, i32 %tile.n, i32 %out.channels %k.short = icmp ult i32 %tile.k, %terms %k.tile = select i1 %k.short, i32 %tile.k, i32 %terms
%m.adjusted = add i32 %m.total, %m.tile %m.numerator = sub i32 %m.adjusted, 1 %m.tiles = udiv i32 %m.numerator, %m.tile %n.adjusted = add i32 %out.channels, %n.tile %n.numerator = sub i32 %n.adjusted, 1 %n.tiles = udiv i32 %n.numerator, %n.tile %jobs = mul i32 %m.tiles, %n.tiles br label %job.loop job.loop:
%job = phi i32 [ %group, %entry ], [ %job.next, %job.done ] %job.more = icmp ult i32 %job, %jobs br i1 %job.more, label %job.step, label %exit job.step:
%m.group.short = icmp ult i32 %m.tiles, RECIPE_CONTRACTION_SWIZZLE_M %m.group.limit = select i1 %m.group.short, i32 %m.tiles, i32 RECIPE_CONTRACTION_SWIZZLE_M %group.width = mul i32 %m.group.limit, %n.tiles %group.index = udiv i32 %job, %group.width %m.group.base = mul i32 %group.index, %m.group.limit %m.group.remaining = sub i32 %m.tiles, %m.group.base %m.group.tail = icmp ult i32 %m.group.remaining, %m.group.limit %m.group.count = select i1 %m.group.tail, i32 %m.group.remaining, i32 %m.group.limit %group.local = urem i32 %job, %group.width %m.group.local = urem i32 %group.local, %m.group.count %m.tile.index = add i32 %m.group.base, %m.group.local %n.tile.index = udiv i32 %group.local, %m.group.count %m.base = mul i32 %m.tile.index, %m.tile %n.base = mul i32 %n.tile.index, %n.tile
%m.remaining = sub i32 %m.total, %m.base %m.partial = icmp ult i32 %m.remaining, %m.tile %m.count = select i1 %m.partial, i32 %m.remaining, i32 %m.tile %n.remaining = sub i32 %out.channels, %n.base %n.partial = icmp ult i32 %n.remaining, %n.tile %n.count = select i1 %n.partial, i32 %n.remaining, i32 %n.tile
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
%method.store = call i1 @contraction_store_lane(i1 %lane.store, i32 %lid)
%lane.n = udiv i32 %output.lane, %m.lanes %lane.m = urem i32 %output.lane, %m.lanes
%output.m.base = mul i32 %lane.m, RECIPE_REGISTER_M %output.n.base = mul i32 %lane.n, RECIPE_REGISTER_N br label %sum.init.loop sum.init.loop:
%sum.init = phi i32 [ 0, %job.step ], [ %sum.init.next, %sum.init.step ] %sum.init.more = icmp ult i32 %sum.init, RECIPE_REGISTER_COUNT br i1 %sum.init.more, label %sum.init.step, label %sum.init.done
sum.init.step: %sum.init.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %sum.init store RECIPE_STATE %state.zero, ptr addrspace(5) %sum.init.ptr, align RECIPE_STATE_ALIGN %sum.init.next = add i32 %sum.init, 1 br label %sum.init.loop sum.init.done: br label %tile.loop tile.loop:
%term.base = phi i32 [ 0, %sum.init.done ], [ %term.next, %tile.done ] %k.remaining = sub i32 %terms, %term.base %k.partial = icmp ult i32 %k.remaining, %k.tile %k.count = select i1 %k.partial, i32 %k.remaining, i32 %k.tile
%a.project = icmp eq i32 %span, 1
%a.unit = icmp eq i32 %in.length, 1
%a.contiguous = and i1 %a.project, %a.unit
%a.fragment.remainder = urem i32 %k.count, RECIPE_FRAGMENT_K
%a.fragment.full = icmp eq i32 %a.fragment.remainder, 0
%a.gate = and i1 %reverse, %relu %a.ungated = xor i1 %a.gate, true %a.vector.shape = and i1 %a.contiguous, %a.fragment.full %a.vector = and i1 %a.vector.shape, %a.ungated
%a.width = select i1 %a.vector, i32 RECIPE_FRAGMENT_K, i32 1
%a.columns = udiv i32 %k.count, %a.width
%b.fragment.remainder = urem i32 %k.count, RECIPE_FRAGMENT_K
%b.fragment.full = icmp eq i32 %b.fragment.remainder, 0 %b.direct = xor i1 %transpose, true %b.vector = and i1 %b.fragment.full, %b.direct
%b.width = select i1 %b.vector, i32 RECIPE_FRAGMENT_K, i32 1
%b.rows = udiv i32 %k.count, %b.width
%a.count = mul i32 %m.count, %a.columns %b.count = mul i32 %n.count, %b.rows %load.count = add i32 %a.count, %b.count br label %load.loop load.loop:
%load = phi i32 [ %lid, %tile.loop ], [ %load.next, %load.advance ] %load.more = icmp ult i32 %load, %load.count br i1 %load.more, label %load.classify, label %load.done load.classify: %load.a = icmp ult i32 %load, %a.count br i1 %load.a, label %load.a.step, label %load.b.step
load.a.step: %a.m = udiv i32 %load, %a.columns %a.column = urem i32 %load, %a.columns %a.k = mul i32 %a.column, %a.width %a.global = add i32 %m.base, %a.m %a.row = udiv i32 %a.global, %out.length %a.position = urem i32 %a.global, %out.length %a.row.base = mul i32 %a.row, %in.elements %a.term = add i32 %term.base, %a.k
%a.tile.index = call i32 @contraction_a_index(i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br i1 %a.vector, label %load.a.vector, label %load.a.scalar
load.a.vector:
%a.vector.index = add i32 %a.row.base, %a.term
%a.vector.source = getelementptr inbounds double, ptr addrspace(1) %input, i32 %a.vector.index
%a.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %a.vector.source, align 8
call void @contraction_stage_a_fragment(<RECIPE_FRAGMENT_K x double> %a.vector.value, i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br label %load.advance
load.a.scalar:
%a.loaded = call double @contraction_input( ptr addrspace(1) %input, i32 %a.row.base, i32 %a.position, i32 %a.term, i32 %span, i32 %in.length, i1 %is.conv )
br i1 %a.gate, label %load.a.activation, label %load.a.ready
load.a.activation:
%a.activation.channel = mul i32 %a.term, %in.length %a.activation.local = add i32 %a.activation.channel, %a.position %a.activation.index = add i32 %a.row.base, %a.activation.local %a.activation.ptr = getelementptr inbounds double, ptr addrspace(1) %activation, i32 %a.activation.index %a.activation.value = load double, ptr addrspace(1) %a.activation.ptr, align 2 %a.activation.positive = call i1 @recipe.ogt(double %a.activation.value, double 0.0) %a.gated = select i1 %a.activation.positive, double %a.loaded, double 0.0
br label %load.a.ready
load.a.ready:
%a.value = phi double [ %a.loaded, %load.a.scalar ], [ %a.gated, %load.a.activation ]
br label %load.store
load.b.step: %b.local = sub i32 %load, %a.count %b.n = udiv i32 %b.local, %b.rows %b.row = urem i32 %b.local, %b.rows %b.k = mul i32 %b.row, %b.width %b.channel = add i32 %n.base, %b.n %b.channel.base = mul i32 %b.channel, %terms %b.term = add i32 %term.base, %b.k
%b.direct.index = add i32 %b.channel.base, %b.term %b.transpose.base = mul i32 %b.term, %out.channels %b.transpose.index = add i32 %b.transpose.base, %b.channel %b.index = select i1 %transpose, i32 %b.transpose.index, i32 %b.direct.index %b.tile.base = mul i32 %tile.m, %tile.k %b.tile.local = call i32 @contraction_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k) %b.tile.index = add i32 %b.tile.base, %b.tile.local
br i1 %b.vector, label %load.b.vector, label %load.b.scalar
load.b.vector:
%b.vector.source = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %b.index
%b.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %b.vector.source, align 8
call void @contraction_stage_b_terms(<RECIPE_FRAGMENT_K x double> %b.vector.value, i32 %b.k, i32 %b.n, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %load.advance
load.b.scalar:
%b.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %b.index
%b.value = load double, ptr addrspace(1) %b.ptr, align 8
br label %load.store
load.store: %load.value = phi double [ %a.value, %load.a.ready ], [ %b.value, %load.b.scalar ] %load.tile.index = phi i32 [ %a.tile.index, %load.a.ready ], [ %b.tile.index, %load.b.scalar ] %load.tile.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %load.tile.index store double %load.value, ptr addrspace(3) %load.tile.ptr, align 8
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
call void @recipe.local.barrier()
call void @contraction_product_accumulate(ptr addrspace(5) %sums, i1 %lane.active, i1 %method.store, i32 %lid, i32 %lane.k, i32 %k.lanes, i32 %output.lane, i32 %lanes, i32 %output.m.base, i32 %output.n.base, i32 %m.count, i32 %n.count, i32 %k.count, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %accumulate.done
accumulate.done:
call void @recipe.local.barrier()
%term.next = add i32 %term.base, %k.count %term.more = icmp ult i32 %term.next, %terms br i1 %term.more, label %tile.done, label %store.loop tile.done: br label %tile.loop store.loop:
%store.register = phi i32 [ 0, %accumulate.done ], [ %store.register.next, %store.next ] %store.more = icmp ult i32 %store.register, RECIPE_REGISTER_COUNT br i1 %store.more, label %store.test, label %job.done
store.test: %store.output.m.raw = call i32 @contraction_output_m(i32 %lid, i32 %store.register, i32 %m.lanes) %store.output.n.raw = call i32 @contraction_output_n(i32 %lid, i32 %store.register, i32 %m.lanes) %store.register.valid = call i1 @contraction_output_register_valid(i32 %store.register)
%store.output.m.valid = icmp ult i32 %store.output.m.raw, %m.count %store.output.n.valid = icmp ult i32 %store.output.n.raw, %n.count %store.output.valid = and i1 %store.output.m.valid, %store.output.n.valid %store.lane.active = and i1 %method.store, %store.output.valid %store.active = and i1 %store.lane.active, %store.register.valid br i1 %store.active, label %store, label %store.next
store: %store.channel = add i32 %n.base, %store.output.n.raw %store.m.global = add i32 %m.base, %store.output.m.raw %store.position = urem i32 %store.m.global, %out.length %store.row = udiv i32 %store.m.global, %out.length %store.output.row.base = mul i32 %store.row, %out.elements
%store.output.channel.base = mul i32 %store.channel, %out.length %store.output.local = add i32 %store.output.channel.base, %store.position %store.output.index = add i32 %store.output.row.base, %store.output.local %store.output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %store.output.index
%store.bias.base = mul i32 %out.channels, %terms %store.bias.index = add i32 %store.bias.base, %store.channel %store.bias.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %store.bias.index %store.bias = load double, ptr addrspace(1) %store.bias.ptr, align 8 %store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %store.register %store.sum.wide = load RECIPE_STATE, ptr addrspace(5) %store.sum.ptr, align RECIPE_STATE_ALIGN %store.sum = call double @recipe.encode(RECIPE_STATE %store.sum.wide)
%store.biased = call double @recipe.add(double %store.sum, double %store.bias) %store.raw = select i1 %has.bias, double %store.biased, double %store.sum %store.forward = xor i1 %reverse, true %store.activate = and i1 %relu, %store.forward %store.positive = call i1 @recipe.ogt(double %store.raw, double 0.0) %store.activated = select i1 %store.positive, double %store.raw, double 0.0 %store.result = select i1 %store.activate, double %store.activated, double %store.raw %store.prior = load double, ptr addrspace(1) %store.output.ptr, align 2 %store.accumulated = call double @recipe.add(double %store.prior, double %store.result) %store.value = select i1 %accumulate, double %store.accumulated, double %store.result store double %store.value, ptr addrspace(1) %store.output.ptr, align 8 br label %store.next
store.next: %store.register.next = add i32 %store.register, 1 br label %store.loop job.done: %job.next = add i32 %job, %groups br label %job.loop exit: ret void }
define internal void @pool_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %context,
i32 %p, i32 %from, i32 %to, i32 %size, i32 %channels ) #1 { entry: %length = udiv i32 %from, %channels
%pooled.length = udiv i32 %to, %channels %row = udiv i32 %p, %to %out = urem i32 %p, %to
%channel = udiv i32 %out, %pooled.length %spatial = urem i32 %out, %pooled.length %start = mul i32 %spatial, %size
%candidate.end = add i32 %start, %size %short = icmp ult i32 %candidate.end, %length
%end = select i1 %short, i32 %candidate.end, i32 %length %row.base = mul i32 %row, %from
%channel.local = mul i32 %channel, %length %input.base = add i32 %row.base, %channel.local br label %loop loop:
%i = phi i32 [ %start, %entry ], [ %next, %step ]
%maximum = phi double [ 0xFFF0000000000000, %entry ], [ %maximum.next, %step ]
%maximum.index = phi i32 [ %start, %entry ], [ %maximum.index.next, %step ] %more = icmp ult i32 %i, %end
br i1 %more, label %step, label %done step: %index = add i32 %input.base, %i
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %index
%value = load double, ptr addrspace(1) %input.ptr, align 8 %greater = call i1 @recipe.ogt(double %value, double %maximum)
%maximum.next = select i1 %greater, double %value, double %maximum
%maximum.index.next = select i1 %greater, i32 %index, i32 %maximum.index %next = add i32 %i, 1 br label %loop done:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %p
%context.ptr = getelementptr inbounds i64, ptr addrspace(1) %context, i32 %p
%maximum.index.wide = zext i32 %maximum.index to i64
store double %maximum, ptr addrspace(1) %output.ptr, align 8
store i64 %maximum.index.wide, ptr addrspace(1) %context.ptr, align 8 ret void }
; Rotary embedding over the first %rotated channels of a fused QKV row: inside
; each head the channel pairs (i, i + dims/2) below %dims rotate by
; position * base^(-2i/dims). With %reverse the transpose rotation is added
; into %output, which makes the same body the adjoint pass.
define internal void @rope_body( ptr addrspace(1) %input, ptr addrspace(1) %output, i32 %p, i32 %channels, i32 %length,
i32 %head.width, i32 %dims, i32 %rotated, double %base, i1 %reverse ) #1 { entry: %per.row = mul i32 %channels, %length
%within = urem i32 %p, %per.row %channel = udiv i32 %within, %length %position = urem i32 %within, %length
%local = urem i32 %channel, %head.width %half = udiv i32 %dims, 2
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %p
%value = load double, ptr addrspace(1) %input.ptr, align 8
%rotates = icmp ult i32 %channel, %rotated %inside = icmp ult i32 %local, %dims %active = and i1 %rotates, %inside
br i1 %active, label %rotate, label %finish rotate: %upper = icmp uge i32 %local, %half
%local.upper = sub i32 %local, %half %index = select i1 %upper, i32 %local.upper, i32 %local
%half.stride = mul i32 %half, %length %partner.up = add i32 %p, %half.stride %partner.down = sub i32 %p, %half.stride
%partner = select i1 %upper, i32 %partner.down, i32 %partner.up
%partner.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %partner
%other = load double, ptr addrspace(1) %partner.ptr, align 8
%two.index = mul i32 %index, 2 %two.index.value = call double @recipe.from.u32(i32 %two.index)
%dims.value = call double @recipe.from.u32(i32 %dims) %ratio = call double @recipe.div(double %two.index.value, double %dims.value)
%log.base = call double @recipe.log(double %base) %exponent.positive = call double @recipe.mul(double %ratio, double %log.base)
%exponent = call double @recipe.neg(double %exponent.positive) %frequency = call double @recipe.exp(double %exponent)
%position.value = call double @recipe.from.u32(i32 %position) %angle = call double @recipe.mul(double %position.value, double %frequency)
%cos = call double @recipe.cos(double %angle) %sin = call double @recipe.sin(double %angle) %sin.negative = call double @recipe.neg(double %sin)
%sin.signed = select i1 %reverse, double %sin.negative, double %sin %sin.signed.negative = call double @recipe.neg(double %sin.signed)
%sin.term = select i1 %upper, double %sin.signed, double %sin.signed.negative
%cos.part = call double @recipe.mul(double %value, double %cos) %sin.part = call double @recipe.mul(double %other, double %sin.term)
%rotated.value = call double @recipe.add(double %cos.part, double %sin.part) br label %finish finish:
%result = phi double [ %value, %entry ], [ %rotated.value, %rotate ]
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %p
br i1 %reverse, label %accumulate, label %assign accumulate: %prior = load double, ptr addrspace(1) %output.ptr, align 8
%sum = call double @recipe.add(double %prior, double %result) store double %sum, ptr addrspace(1) %output.ptr, align 8 ret void
assign: store double %result, ptr addrspace(1) %output.ptr, align 8 ret void }
define internal double @sigmoid(double %x) #1 { entry: %negative = call double @recipe.neg(double %x)
%exponential = call double @recipe.exp(double %negative) %denominator = call double @recipe.add(double 1.0, double %exponential)
%value = call double @recipe.div(double 1.0, double %denominator) ret double %value }
define internal RECIPE_STATE @attention_tile_dot(i32 %left, i32 %right, i32 %width, i32 %left.base, i32 %right.base) #1 { entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %channel.loop
channel.loop:
%channel = phi i32 [ 0, %entry ], [ %channel.next, %channel.step ]
%sum = phi RECIPE_STATE [ %zero, %entry ], [ %sum.next, %channel.step ]
%more = icmp ult i32 %channel, %width
br i1 %more, label %channel.step, label %done
channel.step:
%left.row = mul i32 %left, %width
%left.local = add i32 %left.row, %channel
%left.index = add i32 %left.base, %left.local
%left.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %left.index
%left.value = load double, ptr addrspace(3) %left.ptr, align 8
%right.row = mul i32 %right, %width
%right.local = add i32 %right.row, %channel
%right.index = add i32 %right.base, %right.local
%right.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %right.index
%right.value = load double, ptr addrspace(3) %right.ptr, align 8
%left.wide = call RECIPE_STATE @recipe.decode(double %left.value)
%right.wide = call RECIPE_STATE @recipe.decode(double %right.value)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %left.wide, RECIPE_STATE %right.wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %product)
%channel.next = add i32 %channel, 1
br label %channel.loop
done:
ret RECIPE_STATE %sum
}
define internal double @attention_tile_score(i32 %query, i32 %key, i32 %width, i32 %key.base, double %scale) #1 { entry:
%sum = call RECIPE_STATE @attention_tile_dot(i32 %query, i32 %key, i32 %width, i32 0, i32 %key.base)
%scale.wide = call RECIPE_STATE @recipe.decode(double %scale)
%score = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %sum, RECIPE_STATE %scale.wide)
%result = call double @recipe.encode(RECIPE_STATE %score)
ret double %result
}
; Inverse Euclidean norm of a strided indexer vector, floored by the epsilon.
define internal double @attention_index_scale(ptr addrspace(1) nocapture readonly %input, i32 %base, i32 %count, i32 %stride, double %epsilon) #1 { entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %loop
loop:
%i = phi i32 [ 0, %entry ], [ %i.next, %step ]
%sum = phi RECIPE_STATE [ %zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i32 %i, %count
br i1 %more, label %step, label %done
step:
%offset = mul i32 %i, %stride
%index = add i32 %base, %offset
%ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %index
%value = load double, ptr addrspace(1) %ptr, align 8
%wide = call RECIPE_STATE @recipe.decode(double %value)
%square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %wide, RECIPE_STATE %wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %square)
%i.next = add i32 %i, 1
br label %loop
done:
%epsilon.wide = call RECIPE_STATE @recipe.decode(double %epsilon)
%shifted = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %epsilon.wide)
%root = call RECIPE_STATE @recipe.state.sqrt(RECIPE_STATE %shifted)
%one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%inverse = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %one, RECIPE_STATE %root)
%result = call double @recipe.encode(RECIPE_STATE %inverse)
ret double %result
}
; True when the query keeps the block that holds this key. Each query owns one
; row of block scores followed by one admission flag per block.
define internal i1 @attention_selected(ptr addrspace(1) nocapture readonly %context, i32 %score.row, i32 %blocks, i32 %select.block, i32 %query, i32 %key) #1 { entry:
%stride = mul i32 %blocks, 2
%row = mul i32 %query, %stride
%start = add i32 %score.row, %row
%block.index = udiv i32 %key, %select.block
%flag.row = add i32 %start, %blocks
%flag.index = add i32 %flag.row, %block.index
%flag.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %flag.index
%flag = load double, ptr addrspace(1) %flag.ptr, align 8
%result = call i1 @recipe.ogt(double %flag, double 5.000000e-01)
ret i1 %result
}
; One key-block representative of the indexer: the sum of the unit norm indexer
; keys inside the block.
define internal void @attention_index_body( ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) %context,
i32 %p, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads, i32 %index.heads, i32 %index.width,
i32 %select.block, i1 %gate, double %epsilon ) #1 { entry:
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%kv.planes = mul i32 %kv.plane, 2
%index.query.channels = mul i32 %index.heads, %index.width
%index.channels = add i32 %index.query.channels, %index.width
%index.plane = mul i32 %index.channels, %length
%gate.plane = select i1 %gate, i32 %from, i32 0
%index.query.base = add i32 %from, %kv.planes
%index.key.plane = mul i32 %index.query.channels, %length
%index.key.base = add i32 %index.query.base, %index.key.plane
%row.stride.index = add i32 %index.query.base, %index.plane
%row.stride = add i32 %row.stride.index, %gate.plane
%blocks.numerator = add i32 %length, %select.block
%blocks.less = sub i32 %blocks.numerator, 1
%blocks = udiv i32 %blocks.less, %select.block
%statistics.rows = mul i32 %rows, %heads
%statistics.plane = mul i32 %statistics.rows, %length
%representative.base = mul i32 %statistics.plane, 2
%representative.stride = mul i32 %blocks, %index.width
%row = udiv i32 %p, %blocks
%block.index = urem i32 %p, %blocks
%row.base = mul i32 %row, %row.stride
%key.origin = add i32 %row.base, %index.key.base
%representative.row = mul i32 %row, %representative.stride
%representative.block = mul i32 %block.index, %index.width
%representative.start.row = add i32 %representative.base, %representative.row
%representative.start = add i32 %representative.start.row, %representative.block
%start = mul i32 %block.index, %select.block
%stop.full = add i32 %start, %select.block
%stop.over = icmp ugt i32 %stop.full, %length
%stop = select i1 %stop.over, i32 %length, i32 %stop.full
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%model.zero = call double @recipe.encode(RECIPE_STATE %state.zero)
br label %clear.loop
clear.loop:
%clear.d = phi i32 [ 0, %entry ], [ %clear.next, %clear.step ]
%clear.more = icmp ult i32 %clear.d, %index.width
br i1 %clear.more, label %clear.step, label %key.loop
clear.step:
%clear.index = add i32 %representative.start, %clear.d
%clear.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %clear.index
store double %model.zero, ptr addrspace(1) %clear.ptr, align 8
%clear.next = add i32 %clear.d, 1
br label %clear.loop
key.loop:
%key = phi i32 [ %start, %clear.loop ], [ %key.advance, %key.step ]
%key.more = icmp ult i32 %key, %stop
br i1 %key.more, label %key.prepare, label %exit
key.prepare:
%key.position = add i32 %key.origin, %key
%key.scale = call double @attention_index_scale(ptr addrspace(1) %input, i32 %key.position, i32 %index.width, i32 %length, double %epsilon)
br label %dim.loop
dim.loop:
%dim = phi i32 [ 0, %key.prepare ], [ %dim.advance, %dim.step ]
%dim.more = icmp ult i32 %dim, %index.width
br i1 %dim.more, label %dim.step, label %key.step
dim.step:
%dim.offset = mul i32 %dim, %length
%dim.index = add i32 %key.position, %dim.offset
%dim.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dim.index
%dim.value = load double, ptr addrspace(1) %dim.ptr, align 8
%dim.unit = call double @recipe.mul(double %dim.value, double %key.scale)
%dim.target = add i32 %representative.start, %dim
%dim.target.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %dim.target
%dim.prior = load double, ptr addrspace(1) %dim.target.ptr, align 8
%dim.sum = call double @recipe.add(double %dim.prior, double %dim.unit)
store double %dim.sum, ptr addrspace(1) %dim.target.ptr, align 8
%dim.advance = add i32 %dim, 1
br label %dim.loop
key.step:
%key.advance = add i32 %key, 1
br label %key.loop
exit:
ret void
}
; Block scores and the selection threshold of one query. The indexer scores
; every causal block and the threshold is the score of the keep-th best, so a
; query keeps every block whose score reaches it.
define internal void @attention_select_body( ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) %context,
i32 %p, i32 %keep, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads, i32 %index.heads,
i32 %index.width, i32 %select.block, i1 %gate, double %epsilon ) #1 { entry:
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%kv.planes = mul i32 %kv.plane, 2
%index.query.channels = mul i32 %index.heads, %index.width
%index.channels = add i32 %index.query.channels, %index.width
%index.plane = mul i32 %index.channels, %length
%gate.plane = select i1 %gate, i32 %from, i32 0
%index.query.base = add i32 %from, %kv.planes
%row.stride.index = add i32 %index.query.base, %index.plane
%row.stride = add i32 %row.stride.index, %gate.plane
%blocks.numerator = add i32 %length, %select.block
%blocks.less = sub i32 %blocks.numerator, 1
%blocks = udiv i32 %blocks.less, %select.block
%score.stride = mul i32 %blocks, 2
%statistics.rows = mul i32 %rows, %heads
%statistics.plane = mul i32 %statistics.rows, %length
%representative.base = mul i32 %statistics.plane, 2
%representative.stride = mul i32 %blocks, %index.width
%representative.total = mul i32 %representative.stride, %rows
%score.base = add i32 %representative.base, %representative.total
%row = udiv i32 %p, %length
%query = urem i32 %p, %length
%row.base = mul i32 %row, %row.stride
%query.origin = add i32 %row.base, %index.query.base
%query.position = add i32 %query.origin, %query
%count.less = udiv i32 %query, %select.block
%count = add i32 %count.less, 1
%score.query = mul i32 %p, %score.stride
%score.start = add i32 %score.base, %score.query
%representative.row = mul i32 %row, %representative.stride
%representative.start = add i32 %representative.base, %representative.row
%score.count = mul i32 %rows, %length
%score.plane = mul i32 %score.count, %score.stride
%derivative.base = add i32 %score.base, %score.plane
%derivative.head.stride = mul i32 %length, %blocks
%derivative.row.stride = mul i32 %derivative.head.stride, %heads
%derivative.row = mul i32 %row, %derivative.row.stride
%derivative.query = mul i32 %query, %blocks
%derivative.row.start = add i32 %derivative.base, %derivative.row
%derivative.start = add i32 %derivative.row.start, %derivative.query
%derivative.count = mul i32 %heads, %blocks
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%model.zero = call double @recipe.encode(RECIPE_STATE %state.zero)
br label %clear.loop
clear.loop:
%clear.b = phi i32 [ 0, %entry ], [ %clear.next, %clear.step ]
%clear.more = icmp ult i32 %clear.b, %count
br i1 %clear.more, label %clear.step, label %derivative.clear.loop
clear.step:
%clear.index = add i32 %score.start, %clear.b
%clear.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %clear.index
store double %model.zero, ptr addrspace(1) %clear.ptr, align 8
%clear.next = add i32 %clear.b, 1
br label %clear.loop
derivative.clear.loop:
%derivative.clear.p = phi i32 [ 0, %clear.loop ], [ %derivative.clear.next, %derivative.clear.step ]
%derivative.clear.more = icmp ult i32 %derivative.clear.p, %derivative.count
br i1 %derivative.clear.more, label %derivative.clear.step, label %head.loop
derivative.clear.step:
%derivative.clear.head = udiv i32 %derivative.clear.p, %blocks
%derivative.clear.block = urem i32 %derivative.clear.p, %blocks
%derivative.clear.plane = mul i32 %derivative.clear.head, %derivative.head.stride
%derivative.clear.base = add i32 %derivative.start, %derivative.clear.plane
%derivative.clear.index = add i32 %derivative.clear.base, %derivative.clear.block
%derivative.clear.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %derivative.clear.index
store double %model.zero, ptr addrspace(1) %derivative.clear.ptr, align 8
%derivative.clear.next = add i32 %derivative.clear.p, 1
br label %derivative.clear.loop
head.loop:
%head = phi i32 [ 0, %derivative.clear.loop ], [ %head.advance, %head.done ]
%head.more = icmp ult i32 %head, %index.heads
br i1 %head.more, label %head.prepare, label %threshold.prepare
head.prepare:
%head.offset = mul i32 %head, %index.width
%head.plane = mul i32 %head.offset, %length
%head.base = add i32 %query.position, %head.plane
%head.scale = call double @attention_index_scale(ptr addrspace(1) %input, i32 %head.base, i32 %index.width, i32 %length, double %epsilon)
%head.scale.wide = call RECIPE_STATE @recipe.decode(double %head.scale)
br label %score.loop
score.loop:
%score.b = phi i32 [ 0, %head.prepare ], [ %score.advance, %score.store ]
%score.more = icmp ult i32 %score.b, %count
br i1 %score.more, label %score.prepare, label %head.done
score.prepare:
%score.representative = mul i32 %score.b, %index.width
%score.representative.base = add i32 %representative.start, %score.representative
br label %score.dim.loop
score.dim.loop:
%score.d = phi i32 [ 0, %score.prepare ], [ %score.d.advance, %score.dim.step ]
%score.sum = phi RECIPE_STATE [ %state.zero, %score.prepare ], [ %score.sum.next, %score.dim.step ]
%score.dim.more = icmp ult i32 %score.d, %index.width
br i1 %score.dim.more, label %score.dim.step, label %score.store
score.dim.step:
%score.dim.offset = mul i32 %score.d, %length
%score.query.index = add i32 %head.base, %score.dim.offset
%score.query.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %score.query.index
%score.query.value = load double, ptr addrspace(1) %score.query.ptr, align 8
%score.representative.index = add i32 %score.representative.base, %score.d
%score.representative.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %score.representative.index
%score.representative.value = load double, ptr addrspace(1) %score.representative.ptr, align 8
%score.query.wide = call RECIPE_STATE @recipe.decode(double %score.query.value)
%score.representative.wide = call RECIPE_STATE @recipe.decode(double %score.representative.value)
%score.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %score.query.wide, RECIPE_STATE %score.representative.wide)
%score.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %score.sum, RECIPE_STATE %score.term)
%score.d.advance = add i32 %score.d, 1
br label %score.dim.loop
score.store:
%score.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %score.sum, RECIPE_STATE %head.scale.wide)
%score.index = add i32 %score.start, %score.b
%score.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %score.index
%score.prior = load double, ptr addrspace(1) %score.ptr, align 8
%score.prior.wide = call RECIPE_STATE @recipe.decode(double %score.prior)
%score.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %score.prior.wide, RECIPE_STATE %score.scaled)
%score.value = call double @recipe.encode(RECIPE_STATE %score.total)
store double %score.value, ptr addrspace(1) %score.ptr, align 8
%score.advance = add i32 %score.b, 1
br label %score.loop
head.done:
%head.advance = add i32 %head, 1
br label %head.loop
threshold.prepare:
%flag.base = add i32 %score.start, %blocks
br label %rank.loop
rank.loop:
%rank.b = phi i32 [ 0, %threshold.prepare ], [ %rank.b.next, %rank.store ]
%rank.more = icmp ult i32 %rank.b, %count
br i1 %rank.more, label %rank.body, label %select.exit
rank.body:
%rank.b.index = add i32 %score.start, %rank.b
%rank.b.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %rank.b.index
%rank.b.score = load double, ptr addrspace(1) %rank.b.ptr, align 8
br label %rank.inner
rank.inner:
%rank.c = phi i32 [ 0, %rank.body ], [ %rank.c.next, %rank.inner.step ]
%rank.ahead = phi i32 [ 0, %rank.body ], [ %rank.ahead.next, %rank.inner.step ]
%rank.inner.more = icmp ult i32 %rank.c, %count
br i1 %rank.inner.more, label %rank.inner.step, label %rank.decide
rank.inner.step:
%rank.c.index = add i32 %score.start, %rank.c
%rank.c.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %rank.c.index
%rank.c.score = load double, ptr addrspace(1) %rank.c.ptr, align 8
%rank.greater = call i1 @recipe.ogt(double %rank.c.score, double %rank.b.score)
%rank.same = call i1 @recipe.oeq(double %rank.c.score, double %rank.b.score)
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
%rank.flag = select i1 %rank.admit, double 1.000000e+00, double 0.000000e+00
%rank.flag.index = add i32 %flag.base, %rank.b
%rank.flag.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %rank.flag.index
store double %rank.flag, ptr addrspace(1) %rank.flag.ptr, align 8
%rank.b.next = add i32 %rank.b, 1
br label %rank.loop
select.exit:
ret void
}
; Loss gradient of one block score: the softmax derivative of every key the
; block admits, summed over the attention heads that share the selection.
define internal RECIPE_STATE @attention_index_block_gradient(ptr addrspace(1) nocapture readonly %context, i32 %base, i32 %stride, i32 %heads) #1 { entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %loop
loop:
%head = phi i32 [ 0, %entry ], [ %head.next, %step ]
%sum = phi RECIPE_STATE [ %zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i32 %head, %heads
br i1 %more, label %step, label %done
step:
%offset = mul i32 %head, %stride
%index = add i32 %base, %offset
%ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %index
%value = load double, ptr addrspace(1) %ptr, align 8
%wide = call RECIPE_STATE @recipe.decode(double %value)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %wide)
%head.next = add i32 %head, 1
br label %loop
done:
ret RECIPE_STATE %sum
}
; One channel of the unit indexer query gradient: every causal block
; representative weighted by the gradient of the block score it scored.
define internal RECIPE_STATE @attention_index_query_gradient(ptr addrspace(1) nocapture readonly %context, i32 %derivative.base,
i32 %derivative.stride, i32 %representative.base, i32 %heads, i32 %count, i32 %index.width, i32 %channel) #1 { entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %loop
loop:
%block = phi i32 [ 0, %entry ], [ %block.next, %step ]
%sum = phi RECIPE_STATE [ %zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i32 %block, %count
br i1 %more, label %step, label %done
step:
%slot = add i32 %derivative.base, %block
%gradient = call RECIPE_STATE @attention_index_block_gradient(ptr addrspace(1) %context, i32 %slot, i32 %derivative.stride, i32 %heads)
%representative.offset = mul i32 %block, %index.width
%representative.row = add i32 %representative.base, %representative.offset
%representative.index = add i32 %representative.row, %channel
%representative.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %representative.index
%representative.value = load double, ptr addrspace(1) %representative.ptr, align 8
%representative.wide = call RECIPE_STATE @recipe.decode(double %representative.value)
%term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gradient, RECIPE_STATE %representative.wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %term)
%block.next = add i32 %block, 1
br label %loop
done:
ret RECIPE_STATE %sum
}
; One channel of the unit indexer key gradient. The representative sums the
; unit keys of the block, so every query that scores the block contributes its
; block score gradient times the sum of its unit indexer queries.
define internal RECIPE_STATE @attention_index_key_gradient(ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %context,
i32 %query.origin, i32 %derivative.row, i32 %derivative.stride, i32 %heads, i32 %index.heads, i32 %index.width,
i32 %length, i32 %blocks, i32 %block.index, i32 %start, i32 %channel, double %epsilon ) #1 { entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%channel.offset = mul i32 %channel, %length
br label %query.loop
query.loop:
%query = phi i32 [ %start, %entry ], [ %query.next, %query.step ]
%sum = phi RECIPE_STATE [ %zero, %entry ], [ %sum.next, %query.step ]
%more = icmp ult i32 %query, %length
br i1 %more, label %query.prepare, label %done
query.prepare:
%query.offset = mul i32 %query, %blocks
%query.row = add i32 %derivative.row, %query.offset
%query.slot = add i32 %query.row, %block.index
%gradient = call RECIPE_STATE @attention_index_block_gradient(ptr addrspace(1) %context, i32 %query.slot, i32 %derivative.stride, i32 %heads)
%query.position = add i32 %query.origin, %query
br label %head.loop
head.loop:
%head = phi i32 [ 0, %query.prepare ], [ %head.next, %head.step ]
%unit = phi RECIPE_STATE [ %zero, %query.prepare ], [ %unit.next, %head.step ]
%head.more = icmp ult i32 %head, %index.heads
br i1 %head.more, label %head.step, label %query.step
head.step:
%head.channels = mul i32 %head, %index.width
%head.plane = mul i32 %head.channels, %length
%head.base = add i32 %query.position, %head.plane
%head.scale = call double @attention_index_scale(ptr addrspace(1) %input, i32 %head.base, i32 %index.width, i32 %length, double %epsilon)
%head.index = add i32 %head.base, %channel.offset
%head.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %head.index
%head.value = load double, ptr addrspace(1) %head.ptr, align 8
%head.unit = call double @recipe.mul(double %head.value, double %head.scale)
%head.wide = call RECIPE_STATE @recipe.decode(double %head.unit)
%unit.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %unit, RECIPE_STATE %head.wide)
%head.next = add i32 %head, 1
br label %head.loop
query.step:
%term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gradient, RECIPE_STATE %unit)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %term)
%query.next = add i32 %query, 1
br label %query.loop
done:
ret RECIPE_STATE %sum
}
; Indexer gradient of one sequence position. A block score biases the logits of
; the keys the block admits, so its gradient is the softmax derivative summed
; over those keys, and the chain rule carries it through the unit norms into
; the indexer query planes and the shared indexer key plane.
define internal void @attention_index_reverse_body( ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %context,
ptr addrspace(1) nocapture writeonly %previous, i32 %p, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads,
i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon ) #3 { entry:
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%kv.planes = mul i32 %kv.plane, 2
%index.query.channels = mul i32 %index.heads, %index.width
%index.channels = add i32 %index.query.channels, %index.width
%index.plane = mul i32 %index.channels, %length
%gate.plane = select i1 %gate, i32 %from, i32 0
%index.query.base = add i32 %from, %kv.planes
%index.key.plane = mul i32 %index.query.channels, %length
%index.key.base = add i32 %index.query.base, %index.key.plane
%row.stride.index = add i32 %index.query.base, %index.plane
%row.stride = add i32 %row.stride.index, %gate.plane
%blocks.numerator = add i32 %length, %select.block
%blocks.less = sub i32 %blocks.numerator, 1
%blocks = udiv i32 %blocks.less, %select.block
%score.stride = mul i32 %blocks, 2
%statistics.rows = mul i32 %rows, %heads
%statistics.plane = mul i32 %statistics.rows, %length
%representative.base = mul i32 %statistics.plane, 2
%representative.stride = mul i32 %blocks, %index.width
%representative.total = mul i32 %representative.stride, %rows
%score.base = add i32 %representative.base, %representative.total
%score.count = mul i32 %rows, %length
%score.total = mul i32 %score.count, %score.stride
%derivative.base = add i32 %score.base, %score.total
%derivative.head.stride = mul i32 %length, %blocks
%derivative.row.stride = mul i32 %derivative.head.stride, %heads
%row = udiv i32 %p, %length
%position = urem i32 %p, %length
%row.base = mul i32 %row, %row.stride
%query.origin = add i32 %row.base, %index.query.base
%key.origin = add i32 %row.base, %index.key.base
%representative.row = mul i32 %row, %representative.stride
%representative.start = add i32 %representative.base, %representative.row
%derivative.row = mul i32 %row, %derivative.row.stride
%derivative.row.start = add i32 %derivative.base, %derivative.row
%derivative.query = mul i32 %position, %blocks
%derivative.start = add i32 %derivative.row.start, %derivative.query
%count.less = udiv i32 %position, %select.block
%count = add i32 %count.less, 1
%query.position = add i32 %query.origin, %position
%key.position = add i32 %key.origin, %position
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %query.head.loop
query.head.loop:
%query.head = phi i32 [ 0, %entry ], [ %query.head.next, %query.head.done ]
%query.head.more = icmp ult i32 %query.head, %index.heads
br i1 %query.head.more, label %query.head.prepare, label %key.prepare
query.head.prepare:
%query.head.channels = mul i32 %query.head, %index.width
%query.head.plane = mul i32 %query.head.channels, %length
%query.head.base = add i32 %query.position, %query.head.plane
%query.head.scale = call double @attention_index_scale(ptr addrspace(1) %input, i32 %query.head.base, i32 %index.width, i32 %length, double %epsilon)
%query.head.scale.wide = call RECIPE_STATE @recipe.decode(double %query.head.scale)
br label %query.dot.loop
query.dot.loop:
%query.dot.d = phi i32 [ 0, %query.head.prepare ], [ %query.dot.next, %query.dot.step ]
%query.dot.sum = phi RECIPE_STATE [ %state.zero, %query.head.prepare ], [ %query.dot.sum.next, %query.dot.step ]
%query.dot.more = icmp ult i32 %query.dot.d, %index.width
br i1 %query.dot.more, label %query.dot.step, label %query.store.prepare
query.dot.step:
%query.dot.gradient = call RECIPE_STATE @attention_index_query_gradient(ptr addrspace(1) %context, i32 %derivative.start,
i32 %derivative.head.stride, i32 %representative.start, i32 %heads, i32 %count, i32 %index.width, i32 %query.dot.d)
%query.dot.offset = mul i32 %query.dot.d, %length
%query.dot.index = add i32 %query.head.base, %query.dot.offset
%query.dot.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %query.dot.index
%query.dot.value = load double, ptr addrspace(1) %query.dot.ptr, align 8
%query.dot.wide = call RECIPE_STATE @recipe.decode(double %query.dot.value)
%query.dot.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %query.dot.wide, RECIPE_STATE %query.dot.gradient)
%query.dot.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %query.dot.sum, RECIPE_STATE %query.dot.term)
%query.dot.next = add i32 %query.dot.d, 1
br label %query.dot.loop
query.store.prepare:
%query.dot.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %query.dot.sum, RECIPE_STATE %query.head.scale.wide)
%query.scale.square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %query.head.scale.wide, RECIPE_STATE %query.head.scale.wide)
%query.factor = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %query.dot.scaled, RECIPE_STATE %query.scale.square)
br label %query.store.loop
query.store.loop:
%query.store.d = phi i32 [ 0, %query.store.prepare ], [ %query.store.next, %query.store.step ]
%query.store.more = icmp ult i32 %query.store.d, %index.width
br i1 %query.store.more, label %query.store.step, label %query.head.done
query.store.step:
%query.store.gradient = call RECIPE_STATE @attention_index_query_gradient(ptr addrspace(1) %context, i32 %derivative.start,
i32 %derivative.head.stride, i32 %representative.start, i32 %heads, i32 %count, i32 %index.width, i32 %query.store.d)
%query.store.offset = mul i32 %query.store.d, %length
%query.store.index = add i32 %query.head.base, %query.store.offset
%query.store.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %query.store.index
%query.store.value = load double, ptr addrspace(1) %query.store.ptr, align 8
%query.store.wide = call RECIPE_STATE @recipe.decode(double %query.store.value)
%query.store.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %query.store.gradient, RECIPE_STATE %query.head.scale.wide)
%query.store.projection = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %query.store.wide, RECIPE_STATE %query.factor)
%query.store.total = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %query.store.scaled, RECIPE_STATE %query.store.projection)
%query.store.result = call double @recipe.encode(RECIPE_STATE %query.store.total)
%query.store.target = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %query.store.index
store double %query.store.result, ptr addrspace(1) %query.store.target, align 8
%query.store.next = add i32 %query.store.d, 1
br label %query.store.loop
query.head.done:
%query.head.next = add i32 %query.head, 1
br label %query.head.loop
key.prepare:
%key.block = udiv i32 %position, %select.block
%key.start = mul i32 %key.block, %select.block
%key.scale = call double @attention_index_scale(ptr addrspace(1) %input, i32 %key.position, i32 %index.width, i32 %length, double %epsilon)
%key.scale.wide = call RECIPE_STATE @recipe.decode(double %key.scale)
br label %key.dot.loop
key.dot.loop:
%key.dot.d = phi i32 [ 0, %key.prepare ], [ %key.dot.next, %key.dot.step ]
%key.dot.sum = phi RECIPE_STATE [ %state.zero, %key.prepare ], [ %key.dot.sum.next, %key.dot.step ]
%key.dot.more = icmp ult i32 %key.dot.d, %index.width
br i1 %key.dot.more, label %key.dot.step, label %key.store.prepare
key.dot.step:
%key.dot.gradient = call RECIPE_STATE @attention_index_key_gradient(ptr addrspace(1) %input, ptr addrspace(1) %context,
i32 %query.origin, i32 %derivative.row.start, i32 %derivative.head.stride, i32 %heads, i32 %index.heads, i32 %index.width,
i32 %length, i32 %blocks, i32 %key.block, i32 %key.start, i32 %key.dot.d, double %epsilon)
%key.dot.offset = mul i32 %key.dot.d, %length
%key.dot.index = add i32 %key.position, %key.dot.offset
%key.dot.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %key.dot.index
%key.dot.value = load double, ptr addrspace(1) %key.dot.ptr, align 8
%key.dot.wide = call RECIPE_STATE @recipe.decode(double %key.dot.value)
%key.dot.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %key.dot.wide, RECIPE_STATE %key.dot.gradient)
%key.dot.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %key.dot.sum, RECIPE_STATE %key.dot.term)
%key.dot.next = add i32 %key.dot.d, 1
br label %key.dot.loop
key.store.prepare:
%key.dot.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %key.dot.sum, RECIPE_STATE %key.scale.wide)
%key.scale.square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %key.scale.wide, RECIPE_STATE %key.scale.wide)
%key.factor = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %key.dot.scaled, RECIPE_STATE %key.scale.square)
br label %key.store.loop
key.store.loop:
%key.store.d = phi i32 [ 0, %key.store.prepare ], [ %key.store.next, %key.store.step ]
%key.store.more = icmp ult i32 %key.store.d, %index.width
br i1 %key.store.more, label %key.store.step, label %exit
key.store.step:
%key.store.gradient = call RECIPE_STATE @attention_index_key_gradient(ptr addrspace(1) %input, ptr addrspace(1) %context,
i32 %query.origin, i32 %derivative.row.start, i32 %derivative.head.stride, i32 %heads, i32 %index.heads, i32 %index.width,
i32 %length, i32 %blocks, i32 %key.block, i32 %key.start, i32 %key.store.d, double %epsilon)
%key.store.offset = mul i32 %key.store.d, %length
%key.store.index = add i32 %key.position, %key.store.offset
%key.store.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %key.store.index
%key.store.value = load double, ptr addrspace(1) %key.store.ptr, align 8
%key.store.wide = call RECIPE_STATE @recipe.decode(double %key.store.value)
%key.store.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %key.store.gradient, RECIPE_STATE %key.scale.wide)
%key.store.projection = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %key.store.wide, RECIPE_STATE %key.factor)
%key.store.total = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %key.store.scaled, RECIPE_STATE %key.store.projection)
%key.store.result = call double @recipe.encode(RECIPE_STATE %key.store.total)
%key.store.target = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %key.store.index
store double %key.store.result, ptr addrspace(1) %key.store.target, align 8
%key.store.next = add i32 %key.store.d, 1
br label %key.store.loop
exit:
ret void
}
define internal void @attention_tile_products(ptr addrspace(1) nocapture readonly %output, i32 %output.row,
i32 %delta.base, i32 %product.base, i32 %query.base, i32 %query.count, i32 %head.start,
i32 %head.width, i32 %length, i32 %lid, i32 %block) #1 { entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %query.loop
query.loop:
%query = phi i32 [ %lid, %entry ], [ %query.next, %store ]
%query.more = icmp ult i32 %query, %query.count
br i1 %query.more, label %prepare, label %exit
prepare:
br label %channel.loop
channel.loop:
%channel = phi i32 [ 0, %prepare ], [ %channel.next, %channel.step ]
%sum = phi RECIPE_STATE [ %zero, %prepare ], [ %sum.next, %channel.step ]
%channel.more = icmp ult i32 %channel, %head.width
br i1 %channel.more, label %channel.step, label %store
channel.step:
%shared.row = mul i32 %query, %head.width
%shared.local = add i32 %shared.row, %channel
%delta.index = add i32 %delta.base, %shared.local
%delta.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %delta.index
%delta = load double, ptr addrspace(3) %delta.ptr, align 8
%output.channel = add i32 %head.start, %channel
%output.channel.base = mul i32 %output.channel, %length
%position = add i32 %query.base, %query
%output.local = add i32 %output.channel.base, %position
%output.index = add i32 %output.row, %output.local
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %output.index
%output.value = load double, ptr addrspace(1) %output.ptr, align 8
%delta.wide = call RECIPE_STATE @recipe.decode(double %delta)
%output.wide = call RECIPE_STATE @recipe.decode(double %output.value)
%term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %delta.wide, RECIPE_STATE %output.wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %term)
%channel.next = add i32 %channel, 1
br label %channel.loop
store:
%product.index = add i32 %product.base, %query
%product.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %product.index
%product.value = call double @recipe.encode(RECIPE_STATE %sum)
store double %product.value, ptr addrspace(3) %product.ptr, align 8
%query.next = add i32 %query, %block
br label %query.loop
exit:
ret void
}
define internal void @attention_tile_derivatives(ptr addrspace(1) nocapture readonly %context,
i32 %query.shared, i32 %key.shared, i32 %delta.shared, i32 %value.shared,
i32 %probability.shared, i32 %derivative.shared, i32 %product.shared,
i32 %query.base, i32 %key.base, i32 %query.count, i32 %key.count, i32 %tile.n,
i32 %head.job, i32 %length, i32 %statistics.denominator.base, i32 %head.width,
double %scale, i32 %lid, i32 %block, i32 %score.row, i32 %blocks, i32 %select.block, i1 %select) #1 { entry:
%pair.count = mul i32 %query.count, %key.count
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%model.zero = call double @recipe.encode(RECIPE_STATE %zero)
br label %pair.loop
pair.loop:
%pair = phi i32 [ %lid, %entry ], [ %pair.next, %store ]
%pair.more = icmp ult i32 %pair, %pair.count
br i1 %pair.more, label %prepare, label %exit
prepare:
%query.local = udiv i32 %pair, %key.count
%key.local = urem i32 %pair, %key.count
%query = add i32 %query.base, %query.local
%key = add i32 %key.base, %key.local
%causal = icmp ule i32 %key, %query
br i1 %causal, label %selection, label %invalid
selection:
br i1 %select, label %selection.test, label %complete
selection.test:
%kept = call i1 @attention_selected(ptr addrspace(1) %context, i32 %score.row, i32 %blocks, i32 %select.block, i32 %query, i32 %key)
br i1 %kept, label %complete, label %invalid
complete:
%score.raw = call RECIPE_STATE @attention_tile_dot(i32 %query.local, i32 %key.local, i32 %head.width, i32 %query.shared, i32 %key.shared)
%scale.wide = call RECIPE_STATE @recipe.decode(double %scale)
%score = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %score.raw, RECIPE_STATE %scale.wide)
%dp = call RECIPE_STATE @attention_tile_dot(i32 %query.local, i32 %key.local, i32 %head.width, i32 %delta.shared, i32 %value.shared)
%statistics.base = mul i32 %head.job, %length
%statistics.index = add i32 %statistics.base, %query
%maximum.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %statistics.index
%maximum = load double, ptr addrspace(1) %maximum.ptr, align 8
%denominator.index = add i32 %statistics.denominator.base, %statistics.index
%denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %denominator.index
%denominator = load double, ptr addrspace(1) %denominator.ptr, align 8
%maximum.wide = call RECIPE_STATE @recipe.decode(double %maximum)
%denominator.wide = call RECIPE_STATE @recipe.decode(double %denominator)
%centered = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %score, RECIPE_STATE %maximum.wide)
%exponential = call RECIPE_STATE @recipe.state.exp(RECIPE_STATE %centered)
%probability.wide = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %exponential, RECIPE_STATE %denominator.wide)
%product.index = add i32 %product.shared, %query.local
%product.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %product.index
%product = load double, ptr addrspace(3) %product.ptr, align 8
%product.wide = call RECIPE_STATE @recipe.decode(double %product)
%dp.centered = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %dp, RECIPE_STATE %product.wide)
%derivative.wide = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %probability.wide, RECIPE_STATE %dp.centered)
%probability = call double @recipe.encode(RECIPE_STATE %probability.wide)
%derivative = call double @recipe.encode(RECIPE_STATE %derivative.wide)
br label %store
invalid:
br label %store
store:
%probability.value = phi double [ %probability, %complete ], [ %model.zero, %invalid ]
%derivative.value = phi double [ %derivative, %complete ], [ %model.zero, %invalid ]
%pair.row = mul i32 %query.local, %tile.n
%pair.local = add i32 %pair.row, %key.local
%probability.index = add i32 %probability.shared, %pair.local
%probability.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %probability.index
store double %probability.value, ptr addrspace(3) %probability.ptr, align 8
%derivative.index = add i32 %derivative.shared, %pair.local
%derivative.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %derivative.index
store double %derivative.value, ptr addrspace(3) %derivative.ptr, align 8
%pair.next = add i32 %pair, %block
br label %pair.loop
exit:
ret void
}
define internal void @attention_forward_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %weights,
ptr addrspace(1) nocapture writeonly %output, ptr addrspace(1) %context,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%head.width.double = call double @recipe.from.u32(i32 %head.width)
%scale = call double @recipe.sqrt(double %head.width.double)
%kv.group = udiv i32 %heads, %kv.heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%kv.planes = mul i32 %kv.plane, 2
%value.plane.base = add i32 %from, %kv.plane
%index.query.channels = mul i32 %index.heads, %index.width
%index.channels = add i32 %index.query.channels, %index.width
%index.plane = mul i32 %index.channels, %length
%gate.plane = select i1 %gate, i32 %from, i32 0
%index.query.base = add i32 %from, %kv.planes
%gate.base = add i32 %index.query.base, %index.plane
%row.stride = add i32 %gate.base, %gate.plane
%select = icmp ne i32 %select.block, 0
%block.divisor = select i1 %select, i32 %select.block, i32 1
%blocks.numerator = add i32 %length, %block.divisor
%blocks.less = sub i32 %blocks.numerator, 1
%blocks.full = udiv i32 %blocks.less, %block.divisor
%blocks = select i1 %select, i32 %blocks.full, i32 0
%score.stride = mul i32 %blocks, 2
%tile.m.less.one = sub i32 %tile.m, 1
%query.tiles.rounded = add i32 %length, %tile.m.less.one
%query.tiles = udiv i32 %query.tiles.rounded, %tile.m
%head.jobs = mul i32 %rows, %heads
%statistics.plane = mul i32 %head.jobs, %length
%representative.base = mul i32 %statistics.plane, 2
%representative.stride = mul i32 %blocks, %index.width
%representative.total = mul i32 %representative.stride, %rows
%score.base = add i32 %representative.base, %representative.total
%score.row.stride = mul i32 %length, %score.stride
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
%rescale.base.shared = add i32 %denominator.base.shared, %tile.m
br label %job.loop
job.loop:
%job = phi i32 [ %group, %entry ], [ %job.next, %job.finish ]
%job.more = icmp ult i32 %job, %jobs
br i1 %job.more, label %job.prepare, label %exit
job.prepare:
%query.tile = urem i32 %job, %query.tiles
%head.job = udiv i32 %job, %query.tiles
%head = urem i32 %head.job, %heads
%row = udiv i32 %head.job, %heads
%query.base = mul i32 %query.tile, %tile.m
%query.remaining = sub i32 %length, %query.base
%query.full = icmp ult i32 %query.remaining, %tile.m
%query.count = select i1 %query.full, i32 %query.remaining, i32 %tile.m
%query.last = add i32 %query.base, %query.count
%row.base = mul i32 %row, %row.stride
%head.start = mul i32 %head, %head.width
%kv.head = udiv i32 %head, %kv.group
%kv.head.start = mul i32 %kv.head, %head.width
%score.row = mul i32 %row, %score.row.stride
%score.row.base = add i32 %score.base, %score.row
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
%query.channel.base = mul i32 %query.channel, %length
%query.input.local = add i32 %query.channel.base, %query.position
%query.input.index = add i32 %row.base, %query.input.local
%query.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %query.input.index
%query.value = load double, ptr addrspace(1) %query.input.ptr, align 8
%query.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %query.p
store double %query.value, ptr addrspace(3) %query.shared.ptr, align 8
%accumulator.index = add i32 %accumulator.base.shared, %query.p
%accumulator.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %accumulator.index
store double 0.0, ptr addrspace(3) %accumulator.ptr, align 8
%query.p.next = add i32 %query.p, %block
br label %query.stage.loop
statistics.init.loop:
%statistics.q = phi i32 [ %lid, %query.stage.loop ], [ %statistics.q.next, %statistics.init.step ]
%statistics.q.more = icmp ult i32 %statistics.q, %query.count
br i1 %statistics.q.more, label %statistics.init.step, label %query.stage.done
statistics.init.step:
%maximum.index.init = add i32 %maximum.base.shared, %statistics.q
%maximum.ptr.init = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %maximum.index.init
store double 0xFFF0000000000000, ptr addrspace(3) %maximum.ptr.init, align 8
%denominator.index.init = add i32 %denominator.base.shared, %statistics.q
%denominator.ptr.init = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %denominator.index.init
store double 0.0, ptr addrspace(3) %denominator.ptr.init, align 8
%statistics.q.next = add i32 %statistics.q, %block
br label %statistics.init.loop
query.stage.done:
call void @recipe.local.barrier()
br label %query.norm.done
query.norm.done:
br label %key.tile.loop
key.tile.loop:
%key.tile.base = phi i32 [ 0, %query.norm.done ], [ %key.tile.next, %key.tile.advance ]
%key.tile.more = icmp ult i32 %key.tile.base, %query.last
br i1 %key.tile.more, label %key.tile.prepare, label %output.loop
key.tile.prepare:
%key.remaining = sub i32 %length, %key.tile.base
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
%tile.scan.kept = call i1 @attention_selected(ptr addrspace(1) %context, i32 %score.row.base, i32 %blocks, i32 %select.block, i32 %tile.scan.query.index, i32 %tile.scan.key)
%tile.scan.hit = and i1 %tile.scan.causal, %tile.scan.kept
br i1 %tile.scan.hit, label %key.stage.loop, label %tile.scan.block.advance
tile.scan.block.advance:
%tile.scan.b.next = add i32 %tile.scan.b, 1
br label %tile.scan.block.loop
tile.scan.block.done:
%tile.scan.q.next = add i32 %tile.scan.q, 1
br label %tile.scan.loop
key.stage.loop:
%key.p = phi i32 [ %lid, %key.tile.prepare ], [ %lid, %tile.scan.block.step ], [ %key.p.next, %key.stage.step ]
%key.p.more = icmp ult i32 %key.p, %active.key.values
br i1 %key.p.more, label %key.stage.step, label %key.stage.done
key.stage.step:
%key.local = udiv i32 %key.p, %head.width
%key.channel.local = urem i32 %key.p, %head.width
%key.position = add i32 %key.tile.base, %key.local
%key.channel = add i32 %kv.head.start, %key.channel.local
%key.channel.base = mul i32 %key.channel, %length
%key.input.local = add i32 %key.channel.base, %key.position
%key.plane = add i32 %row.base, %from
%key.input.index = add i32 %key.plane, %key.input.local
%key.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %key.input.index
%key.value = load double, ptr addrspace(1) %key.input.ptr, align 8
%key.shared.index = add i32 %key.base.shared, %key.p
%key.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %key.shared.index
store double %key.value, ptr addrspace(3) %key.shared.ptr, align 8
%value.row = add i32 %row.base, %value.plane.base
%value.input.index = add i32 %value.row, %key.input.local
%value.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %value.input.index
%value.value = load double, ptr addrspace(1) %value.input.ptr, align 8
%value.shared.index = add i32 %value.base.shared, %key.p
%value.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %value.shared.index
store double %value.value, ptr addrspace(3) %value.shared.ptr, align 8
%key.p.next = add i32 %key.p, %block
br label %key.stage.loop
key.stage.done:
call void @recipe.local.barrier()
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
br i1 %select, label %score.selection.test, label %score.complete
score.selection.test:
%score.kept = call i1 @attention_selected(ptr addrspace(1) %context, i32 %score.row.base, i32 %blocks, i32 %select.block, i32 %score.query, i32 %score.key)
br i1 %score.kept, label %score.complete, label %score.invalid
score.complete:
%score.scaled = call double @attention_tile_score(i32 %score.query.local, i32 %score.key.local, i32 %head.width, i32 %key.base.shared, double %scale)
br label %score.store
score.invalid:
br label %score.store
score.store:
%score.value = phi double [ %score.scaled, %score.complete ], [ 0xFFF0000000000000, %score.invalid ]
%score.shared.row = mul i32 %score.query.local, %tile.n
%score.shared.local = add i32 %score.shared.row, %score.key.local
%score.shared.index = add i32 %score.base.shared, %score.shared.local
%score.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %score.shared.index
store double %score.value, ptr addrspace(3) %score.shared.ptr, align 8
%score.p.next = add i32 %score.p, %block
br label %score.loop
score.done:
call void @recipe.local.barrier()
br label %softmax.loop
softmax.loop:
%softmax.query = phi i32 [ %lid, %score.done ], [ %softmax.query.next, %softmax.store ]
%softmax.more = icmp ult i32 %softmax.query, %query.count
br i1 %softmax.more, label %maximum.load, label %softmax.done
maximum.load:
%maximum.index = add i32 %maximum.base.shared, %softmax.query
%maximum.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %maximum.index
%maximum.old = load double, ptr addrspace(3) %maximum.ptr, align 8
br label %maximum.loop
maximum.loop:
%maximum.key = phi i32 [ 0, %maximum.load ], [ %maximum.key.next, %maximum.step ]
%maximum.value = phi double [ %maximum.old, %maximum.load ], [ %maximum.next, %maximum.step ]
%maximum.more = icmp ult i32 %maximum.key, %key.count
br i1 %maximum.more, label %maximum.step, label %probability.prepare
maximum.step:
%maximum.score.row = mul i32 %softmax.query, %tile.n
%maximum.score.local = add i32 %maximum.score.row, %maximum.key
%maximum.score.index = add i32 %score.base.shared, %maximum.score.local
%maximum.score.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %maximum.score.index
%maximum.score = load double, ptr addrspace(3) %maximum.score.ptr, align 8
%maximum.larger = call i1 @recipe.ogt(double %maximum.score, double %maximum.value)
%maximum.next = select i1 %maximum.larger, double %maximum.score, double %maximum.value
%maximum.key.next = add i32 %maximum.key, 1
br label %maximum.loop
probability.prepare:
%denominator.index = add i32 %denominator.base.shared, %softmax.query
%denominator.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %denominator.index
%denominator.old = load double, ptr addrspace(3) %denominator.ptr, align 8
%maximum.old.centered = call double @recipe.sub(double %maximum.old, double %maximum.value)
%old.rescale = call double @recipe.exp(double %maximum.old.centered)
%denominator.old.wide = call RECIPE_STATE @recipe.decode(double %denominator.old)
%old.rescale.wide = call RECIPE_STATE @recipe.decode(double %old.rescale)
%denominator.rescaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %denominator.old.wide, RECIPE_STATE %old.rescale.wide)
br label %probability.loop
probability.loop:
%probability.key = phi i32 [ 0, %probability.prepare ], [ %probability.key.next, %probability.step ]
%denominator.value = phi RECIPE_STATE [ %denominator.rescaled, %probability.prepare ], [ %denominator.next, %probability.step ]
%probability.more = icmp ult i32 %probability.key, %key.count
br i1 %probability.more, label %probability.step, label %softmax.store
probability.step:
%probability.row = mul i32 %softmax.query, %tile.n
%probability.local = add i32 %probability.row, %probability.key
%probability.score.index = add i32 %score.base.shared, %probability.local
%probability.score.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %probability.score.index
%probability.score = load double, ptr addrspace(3) %probability.score.ptr, align 8
%probability.centered = call double @recipe.sub(double %probability.score, double %maximum.value)
%probability.value = call double @recipe.exp(double %probability.centered)
%probability.index = add i32 %probability.base.shared, %probability.local
%probability.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %probability.index
store double %probability.value, ptr addrspace(3) %probability.ptr, align 8
%probability.wide = call RECIPE_STATE @recipe.decode(double %probability.value)
%denominator.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %denominator.value, RECIPE_STATE %probability.wide)
%probability.key.next = add i32 %probability.key, 1
br label %probability.loop
softmax.store:
%denominator.model = call double @recipe.encode(RECIPE_STATE %denominator.value)
store double %maximum.value, ptr addrspace(3) %maximum.ptr, align 8
store double %denominator.model, ptr addrspace(3) %denominator.ptr, align 8
%rescale.index = add i32 %rescale.base.shared, %softmax.query
%rescale.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %rescale.index
store double %old.rescale, ptr addrspace(3) %rescale.ptr, align 8
%softmax.query.next = add i32 %softmax.query, %block
br label %softmax.loop
softmax.done:
call void @recipe.local.barrier()
br label %accumulate.loop
accumulate.loop:
%accumulate.p = phi i32 [ %lid, %softmax.done ], [ %accumulate.p.next, %accumulate.store ]
%accumulate.more = icmp ult i32 %accumulate.p, %active.query.values
br i1 %accumulate.more, label %accumulate.prepare, label %accumulate.done
accumulate.prepare:
%accumulate.query = udiv i32 %accumulate.p, %head.width
%accumulate.channel = urem i32 %accumulate.p, %head.width
%accumulate.index = add i32 %accumulator.base.shared, %accumulate.p
%accumulate.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %accumulate.index
%accumulate.old = load double, ptr addrspace(3) %accumulate.ptr, align 8
%accumulate.rescale.index = add i32 %rescale.base.shared, %accumulate.query
%accumulate.rescale.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %accumulate.rescale.index
%accumulate.rescale = load double, ptr addrspace(3) %accumulate.rescale.ptr, align 8
%accumulate.initial = call double @recipe.mul(double %accumulate.old, double %accumulate.rescale)
br label %accumulate.key.loop
accumulate.key.loop:
%accumulate.key = phi i32 [ 0, %accumulate.prepare ], [ %accumulate.key.next, %accumulate.key.step ]
%accumulate.value = phi double [ %accumulate.initial, %accumulate.prepare ], [ %accumulate.next, %accumulate.key.step ]
%accumulate.key.more = icmp ult i32 %accumulate.key, %key.count
br i1 %accumulate.key.more, label %accumulate.key.step, label %accumulate.store
accumulate.key.step:
%accumulate.probability.row = mul i32 %accumulate.query, %tile.n
%accumulate.probability.local = add i32 %accumulate.probability.row, %accumulate.key
%accumulate.probability.index = add i32 %probability.base.shared, %accumulate.probability.local
%accumulate.probability.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %accumulate.probability.index
%accumulate.probability = load double, ptr addrspace(3) %accumulate.probability.ptr, align 8
%accumulate.value.row = mul i32 %accumulate.key, %head.width
%accumulate.value.local = add i32 %accumulate.value.row, %accumulate.channel
%accumulate.value.index = add i32 %value.base.shared, %accumulate.value.local
%accumulate.value.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %accumulate.value.index
%accumulate.v = load double, ptr addrspace(3) %accumulate.value.ptr, align 8
%accumulate.weighted = call double @recipe.mul(double %accumulate.probability, double %accumulate.v)
%accumulate.next = call double @recipe.add(double %accumulate.value, double %accumulate.weighted)
%accumulate.key.next = add i32 %accumulate.key, 1
br label %accumulate.key.loop
accumulate.store:
store double %accumulate.value, ptr addrspace(3) %accumulate.ptr, align 8
%accumulate.p.next = add i32 %accumulate.p, %block
br label %accumulate.loop
accumulate.done:
call void @recipe.local.barrier()
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
%output.accumulator.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %output.accumulator.index
%output.accumulator = load double, ptr addrspace(3) %output.accumulator.ptr, align 8
%output.denominator.index = add i32 %denominator.base.shared, %output.query.local
%output.denominator.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %output.denominator.index
%output.denominator = load double, ptr addrspace(3) %output.denominator.ptr, align 8
%attention = call double @recipe.div(double %output.accumulator, double %output.denominator)
%output.query = add i32 %query.base, %output.query.local
%output.statistics.owner = icmp eq i32 %output.channel.local, 0
br i1 %output.statistics.owner, label %output.statistics.store, label %output.value.store
output.statistics.store:
%output.maximum.index = add i32 %maximum.base.shared, %output.query.local
%output.maximum.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %output.maximum.index
%output.maximum = load double, ptr addrspace(3) %output.maximum.ptr, align 8
%output.statistics.head.base = mul i32 %head.job, %length
%output.statistics.index = add i32 %output.statistics.head.base, %output.query
%output.statistics.maximum.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %output.statistics.index
store double %output.maximum, ptr addrspace(1) %output.statistics.maximum.ptr, align 8
%output.statistics.denominator.index = add i32 %statistics.plane, %output.statistics.index
%output.statistics.denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %output.statistics.denominator.index
store double %output.denominator, ptr addrspace(1) %output.statistics.denominator.ptr, align 8
br label %output.value.store
output.value.store:
%output.channel = add i32 %head.start, %output.channel.local
%output.channel.base = mul i32 %output.channel, %length
%output.local = add i32 %output.channel.base, %output.query
%output.row.base = mul i32 %row, %from
%output.index = add i32 %output.row.base, %output.local
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %output.index
br i1 %gate, label %output.gate, label %output.plain
output.gate:
%output.gate.row = add i32 %row.base, %gate.base
%output.gate.index = add i32 %output.gate.row, %output.local
%output.gate.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %output.gate.index
%output.gate.value = load double, ptr addrspace(1) %output.gate.ptr, align 8
%output.gate.factor = call double @recipe.sigmoid(double %output.gate.value)
%output.gated = call double @recipe.mul(double %attention, double %output.gate.factor)
br label %output.plain
output.plain:
%output.result = phi double [ %attention, %output.value.store ], [ %output.gated, %output.gate ]
store double %output.result, ptr addrspace(1) %output.ptr, align 8
%output.p.next = add i32 %output.p, %block
br label %output.loop
output.done:
call void @recipe.local.barrier()
br label %job.finish
job.finish:
%job.next = add i32 %job, %groups
br label %job.loop
exit:
ret void
}
define internal void @attention_forward_matrix_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %weights,
ptr addrspace(1) nocapture writeonly %output, ptr addrspace(1) %context,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%width.double = call double @recipe.from.u32(i32 %head.width)
%scale = call double @recipe.sqrt(double %width.double)
%head.jobs = mul i32 %rows, %heads
%statistics.rows = mul i32 %head.jobs, %length
%head.values = mul i32 %length, %head.width
%pair.values = mul i32 %length, %length
%q.base = add i32 0, 0
%k.base = add i32 %q.base, %head.values
%v.base = add i32 %k.base, %head.values
%p.base = add i32 %v.base, %head.values
br label %attention.forward.matrix.job.loop
attention.forward.matrix.job.loop:
%head.job = phi i32 [ %group, %entry ], [ %head.job.next, %attention.forward.matrix.job.done ]
%head.job.more = icmp ult i32 %head.job, %head.jobs
br i1 %head.job.more, label %attention.forward.matrix.job.step, label %attention.forward.matrix.exit
attention.forward.matrix.job.step:
%head = urem i32 %head.job, %heads
%row = udiv i32 %head.job, %heads
%head.start = mul i32 %head, %head.width
%input.row.stride = mul i32 %from, 3
%input.row = mul i32 %row, %input.row.stride
%output.row = mul i32 %row, %from
br label %attention.forward.matrix.stage.channel.loop
attention.forward.matrix.stage.channel.loop:
%stage.channel.local = phi i32 [ %lid, %attention.forward.matrix.job.step ], [ %stage.channel.next, %attention.forward.matrix.stage.channel.done ]
%stage.channel.more = icmp ult i32 %stage.channel.local, %head.width
br i1 %stage.channel.more, label %attention.forward.matrix.stage.channel.step, label %attention.forward.matrix.stage.done
attention.forward.matrix.stage.channel.step:
%stage.channel = add i32 %head.start, %stage.channel.local
%stage.channel.base = mul i32 %stage.channel, %length
br label %attention.forward.matrix.stage.plane.loop
attention.forward.matrix.stage.plane.loop:
%stage.plane = phi i32 [ 0, %attention.forward.matrix.stage.channel.step ], [ %stage.plane.next, %attention.forward.matrix.stage.plane.done ]
%stage.plane.more = icmp ult i32 %stage.plane, 3
br i1 %stage.plane.more, label %attention.forward.matrix.stage.plane.step, label %attention.forward.matrix.stage.channel.done
attention.forward.matrix.stage.plane.step:
%stage.input.plane = mul i32 %stage.plane, %from
%stage.input.row = add i32 %input.row, %stage.input.plane
%stage.input.base = add i32 %stage.input.row, %stage.channel.base
%stage.shared.base = mul i32 %stage.plane, %head.values
br label %attention.forward.matrix.stage.position.loop
attention.forward.matrix.stage.position.loop:
%stage.position = phi i32 [ 0, %attention.forward.matrix.stage.plane.step ], [ %stage.position.next, %attention.forward.matrix.stage.position.advance ]
%stage.position.more = icmp ult i32 %stage.position, %length
br i1 %stage.position.more, label %attention.forward.matrix.stage.vector.test, label %attention.forward.matrix.stage.plane.done
attention.forward.matrix.stage.vector.test:
%stage.position.remaining = sub i32 %length, %stage.position
%stage.vector = icmp uge i32 %stage.position.remaining, 16
br i1 %stage.vector, label %attention.forward.matrix.stage.vector, label %attention.forward.matrix.stage.scalar
attention.forward.matrix.stage.vector:
%stage.vector.index = add i32 %stage.input.base, %stage.position
%stage.vector.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %stage.vector.index
%stage.vector.value = load <16 x double>, ptr addrspace(1) %stage.vector.ptr, align 8
call void @contraction_stage_column16(<16 x double> %stage.vector.value, i32 %stage.shared.base, i32 %stage.position, i32 %stage.channel.local, i32 %head.width)
%stage.vector.position.next = add i32 %stage.position, 16
br label %attention.forward.matrix.stage.position.advance
attention.forward.matrix.stage.scalar:
%stage.scalar.index = add i32 %stage.input.base, %stage.position
%stage.scalar.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %stage.scalar.index
%stage.scalar.value = load double, ptr addrspace(1) %stage.scalar.ptr, align 8
%stage.scalar.row = mul i32 %stage.position, %head.width
%stage.scalar.local = add i32 %stage.scalar.row, %stage.channel.local
%stage.scalar.index.shared = add i32 %stage.shared.base, %stage.scalar.local
%stage.scalar.ptr.shared = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %stage.scalar.index.shared
store double %stage.scalar.value, ptr addrspace(3) %stage.scalar.ptr.shared, align 8
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
call void @recipe.local.barrier()
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
%matrix.accumulator = phi <8 x RECIPE_STATE> [ zeroinitializer, %attention.forward.matrix.score.job.step ], [ %matrix.accumulator.next, %attention.forward.matrix.score.width.step ]
%matrix.width.more = icmp ult i32 %matrix.width, %head.width
br i1 %matrix.width.more, label %attention.forward.matrix.score.width.step, label %attention.forward.matrix.score.store.loop
attention.forward.matrix.score.width.step:
%matrix.q.row = mul i32 %matrix.q.safe, %head.width
%matrix.q.local = add i32 %matrix.q.row, %matrix.width
%matrix.q.index = add i32 %q.base, %matrix.q.local
%matrix.q.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.q.index
%matrix.q.fragment = load <16 x double>, ptr addrspace(3) %matrix.q.ptr, align 2
%matrix.k.row = mul i32 %matrix.k.safe, %head.width
%matrix.k.local = add i32 %matrix.k.row, %matrix.width
%matrix.k.index = add i32 %k.base, %matrix.k.local
%matrix.k.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.k.index
%matrix.k.fragment = load <16 x double>, ptr addrspace(3) %matrix.k.ptr, align 2
%matrix.accumulator.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %matrix.q.fragment, <16 x double> %matrix.k.fragment, <8 x RECIPE_STATE> %matrix.accumulator)
%matrix.width.next = add i32 %matrix.width, 16
br label %attention.forward.matrix.score.width.loop
attention.forward.matrix.score.store.loop:
%matrix.output = phi i32 [ 0, %attention.forward.matrix.score.width.loop ], [ %matrix.output.next, %attention.forward.matrix.score.store.next ]
%matrix.output.more = icmp ult i32 %matrix.output, 8
br i1 %matrix.output.more, label %attention.forward.matrix.score.store.test, label %attention.forward.matrix.score.store.done
attention.forward.matrix.score.store.test:
%matrix.output.twice = mul i32 %matrix.output, 2
%matrix.query.local = add i32 %matrix.output.twice, %matrix.lane.half
%matrix.query = add i32 %matrix.q.tile, %matrix.query.local
%matrix.query.valid = icmp ult i32 %matrix.query, %length
%matrix.pair.valid = and i1 %matrix.query.valid, %matrix.k.valid
br i1 %matrix.pair.valid, label %attention.forward.matrix.score.store, label %attention.forward.matrix.score.store.next
attention.forward.matrix.score.store:
%matrix.score.wide = extractelement <8 x RECIPE_STATE> %matrix.accumulator, i32 %matrix.output
%matrix.score.raw = call double @recipe.encode(RECIPE_STATE %matrix.score.wide)
%matrix.score.scaled = call double @recipe.div(double %matrix.score.raw, double %scale)
%matrix.causal = icmp ule i32 %matrix.k, %matrix.query
%matrix.score = select i1 %matrix.causal, double %matrix.score.scaled, double 0xFFF0000000000000
%matrix.pair.row = mul i32 %matrix.query, %length
%matrix.pair.local = add i32 %matrix.pair.row, %matrix.k
%matrix.p.index = add i32 %p.base, %matrix.pair.local
%matrix.p.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.p.index
store double %matrix.score, ptr addrspace(3) %matrix.p.ptr, align 8
br label %attention.forward.matrix.score.store.next
attention.forward.matrix.score.store.next:
%matrix.output.next = add i32 %matrix.output, 1
br label %attention.forward.matrix.score.store.loop
attention.forward.matrix.score.store.done:
%matrix.job.next = add i32 %matrix.job, %matrix.waves
br label %attention.forward.matrix.score.job.loop
attention.forward.matrix.score.done:
call void @recipe.local.barrier()
br label %attention.forward.matrix.softmax.loop
attention.forward.matrix.softmax.loop:
%softmax.query = phi i32 [ %lid, %attention.forward.matrix.score.done ], [ %softmax.query.next, %attention.forward.matrix.softmax.store ]
%softmax.more = icmp ult i32 %softmax.query, %length
br i1 %softmax.more, label %attention.forward.matrix.maximum.loop, label %attention.forward.matrix.softmax.done
attention.forward.matrix.maximum.loop:
%maximum.key = phi i32 [ 0, %attention.forward.matrix.softmax.loop ], [ %maximum.key.next, %attention.forward.matrix.maximum.step ]
%maximum = phi double [ 0xFFF0000000000000, %attention.forward.matrix.softmax.loop ], [ %maximum.next, %attention.forward.matrix.maximum.step ]
%maximum.more = icmp ult i32 %maximum.key, %length
br i1 %maximum.more, label %attention.forward.matrix.maximum.step, label %attention.forward.matrix.probability.loop
attention.forward.matrix.maximum.step:
%maximum.row = mul i32 %softmax.query, %length
%maximum.local = add i32 %maximum.row, %maximum.key
%maximum.index = add i32 %p.base, %maximum.local
%maximum.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %maximum.index
%maximum.score = load double, ptr addrspace(3) %maximum.ptr, align 8
%maximum.larger = call i1 @recipe.ogt(double %maximum.score, double %maximum)
%maximum.next = select i1 %maximum.larger, double %maximum.score, double %maximum
%maximum.key.next = add i32 %maximum.key, 1
br label %attention.forward.matrix.maximum.loop
attention.forward.matrix.probability.loop:
%probability.key = phi i32 [ 0, %attention.forward.matrix.maximum.loop ], [ %probability.key.next, %attention.forward.matrix.probability.step ]
%denominator = phi double [ 0.0, %attention.forward.matrix.maximum.loop ], [ %denominator.next, %attention.forward.matrix.probability.step ]
%probability.more = icmp ult i32 %probability.key, %length
br i1 %probability.more, label %attention.forward.matrix.probability.step, label %attention.forward.matrix.normalize.loop
attention.forward.matrix.probability.step:
%probability.row = mul i32 %softmax.query, %length
%probability.local = add i32 %probability.row, %probability.key
%probability.index = add i32 %p.base, %probability.local
%probability.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %probability.index
%probability.score = load double, ptr addrspace(3) %probability.ptr, align 8
%probability.centered = call double @recipe.sub(double %probability.score, double %maximum)
%probability.value = call double @recipe.exp(double %probability.centered)
store double %probability.value, ptr addrspace(3) %probability.ptr, align 8
%denominator.next = call double @recipe.add(double %denominator, double %probability.value)
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
%normalize.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %normalize.index
%normalize.value = load double, ptr addrspace(3) %normalize.ptr, align 8
%normalized = call double @recipe.div(double %normalize.value, double %denominator)
store double %normalized, ptr addrspace(3) %normalize.ptr, align 8
%normalize.key.next = add i32 %normalize.key, 1
br label %attention.forward.matrix.normalize.loop
attention.forward.matrix.softmax.store:
%statistics.base = mul i32 %head.job, %length
%statistics.index = add i32 %statistics.base, %softmax.query
%maximum.context.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %statistics.index
store double %maximum, ptr addrspace(1) %maximum.context.ptr, align 8
%denominator.index = add i32 %statistics.rows, %statistics.index
%denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %denominator.index
store double %denominator, ptr addrspace(1) %denominator.ptr, align 8
%softmax.query.next = add i32 %softmax.query, %block
br label %attention.forward.matrix.softmax.loop
attention.forward.matrix.softmax.done:
call void @recipe.local.barrier()
call void @attention_matrix_product(ptr addrspace(1) %output, i32 3, i32 %p.base, i32 %v.base, i32 %output.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
call void @recipe.local.barrier()
br label %attention.forward.matrix.job.done
attention.forward.matrix.job.done:
%head.job.next = add i32 %head.job, %groups
br label %attention.forward.matrix.job.loop
attention.forward.matrix.exit:
ret void
}
define internal void @attention_matrix_product(
ptr addrspace(1) %previous, i32 %mode, i32 %left.base, i32 %right.base,
i32 %row.base, i32 %from, i32 %head.start, i32 %length, i32 %head.width,
double %scale, i32 %lid, i32 %block ) #1 { entry:
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
%accumulator = phi <8 x RECIPE_STATE> [ zeroinitializer, %attention.matrix.gradient.job.step ], [ %accumulator.next, %attention.matrix.gradient.fragment.done ]
%k.more = icmp ult i32 %k.base, %length
br i1 %k.more, label %attention.matrix.gradient.fragment.loop, label %attention.matrix.gradient.store.loop
attention.matrix.gradient.fragment.loop:
%fragment = phi i32 [ 0, %attention.matrix.gradient.k.loop ], [ %fragment.next, %attention.matrix.gradient.fragment.step ]
%left.fragment = phi <16 x double> [ zeroinitializer, %attention.matrix.gradient.k.loop ], [ %left.fragment.next, %attention.matrix.gradient.fragment.step ]
%right.fragment = phi <16 x double> [ zeroinitializer, %attention.matrix.gradient.k.loop ], [ %right.fragment.next, %attention.matrix.gradient.fragment.step ]
%fragment.more = icmp ult i32 %fragment, 16
br i1 %fragment.more, label %attention.matrix.gradient.fragment.step, label %attention.matrix.gradient.fragment.done
attention.matrix.gradient.fragment.step:
%term = add i32 %k.base, %fragment
%term.valid = icmp ult i32 %term, %length
%term.safe = select i1 %term.valid, i32 %term, i32 0
%left.direct.row = mul i32 %m.safe, %length
%left.direct.local = add i32 %left.direct.row, %term.safe
%left.transpose.row = mul i32 %term.safe, %length
%left.transpose.local = add i32 %left.transpose.row, %m.safe
%left.local = select i1 %direct, i32 %left.direct.local, i32 %left.transpose.local
%left.index = add i32 %left.base, %left.local
%left.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %left.index
%left.loaded = load double, ptr addrspace(3) %left.ptr, align 8
%right.row = mul i32 %term.safe, %head.width
%right.local = add i32 %right.row, %n.safe
%right.index = add i32 %right.base, %right.local
%right.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %right.index
%right.loaded = load double, ptr addrspace(3) %right.ptr, align 8
%left.valid = and i1 %m.valid, %term.valid
%right.valid = and i1 %n.valid, %term.valid
%left.value = select i1 %left.valid, double %left.loaded, double 0.0
%right.value = select i1 %right.valid, double %right.loaded, double 0.0
%left.fragment.next = insertelement <16 x double> %left.fragment, double %left.value, i32 %fragment
%right.fragment.next = insertelement <16 x double> %right.fragment, double %right.value, i32 %fragment
%fragment.next = add i32 %fragment, 1
br label %attention.matrix.gradient.fragment.loop
attention.matrix.gradient.fragment.done:
%accumulator.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %left.fragment, <16 x double> %right.fragment, <8 x RECIPE_STATE> %accumulator)
%k.next = add i32 %k.base, 16
br label %attention.matrix.gradient.k.loop
attention.matrix.gradient.store.loop:
%output = phi i32 [ 0, %attention.matrix.gradient.k.loop ], [ %output.next, %attention.matrix.gradient.store.next ]
%output.more = icmp ult i32 %output, 8
br i1 %output.more, label %attention.matrix.gradient.store.test, label %attention.matrix.gradient.store.done
attention.matrix.gradient.store.test:
%output.twice = mul i32 %output, 2
%output.m.local = add i32 %output.twice, %lane.half
%output.m = add i32 %m.tile, %output.m.local
%output.m.valid = icmp ult i32 %output.m, %length
%output.valid = and i1 %output.m.valid, %n.valid
br i1 %output.valid, label %attention.matrix.gradient.store, label %attention.matrix.gradient.store.next
attention.matrix.gradient.store:
%output.wide = extractelement <8 x RECIPE_STATE> %accumulator, i32 %output
%output.raw = call double @recipe.encode(RECIPE_STATE %output.wide)
%output.scaled = call double @recipe.div(double %output.raw, double %scale)
%output.value = select i1 %unscaled, double %output.raw, double %output.scaled
%output.plane.raw = mul i32 %mode, %from
%output.plane = select i1 %forward, i32 0, i32 %output.plane.raw
%output.row = add i32 %row.base, %output.plane
%output.channel = add i32 %head.start, %n
%output.channel.base = mul i32 %output.channel, %length
%output.local = add i32 %output.channel.base, %output.m
%output.index = add i32 %output.row, %output.local
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %output.index
store double %output.value, ptr addrspace(1) %output.ptr, align 8
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
i32 %kv.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%width.double = call double @recipe.from.u32(i32 %head.width)
%scale = call double @recipe.sqrt(double %width.double)
%head.jobs = mul i32 %rows, %heads
%statistics.rows = mul i32 %head.jobs, %length
%head.values = mul i32 %length, %head.width
%pair.values = mul i32 %length, %length
%q.base = add i32 0, 0
%k.base = add i32 %q.base, %head.values
%v.base = add i32 %k.base, %head.values
%do.base = add i32 %v.base, %head.values
%p.base = add i32 %do.base, %head.values
%ds.base = add i32 %p.base, %pair.values
%d.base = add i32 %ds.base, %pair.values
br label %attention.matrix.job.loop
attention.matrix.job.loop:
%head.job = phi i32 [ %group, %entry ], [ %head.job.next, %attention.matrix.job.done ]
%head.job.more = icmp ult i32 %head.job, %head.jobs
br i1 %head.job.more, label %attention.matrix.job.step, label %attention.matrix.exit
attention.matrix.job.step:
%head = urem i32 %head.job, %heads
%row = udiv i32 %head.job, %heads
%head.start = mul i32 %head, %head.width
%input.row.stride = mul i32 %from, 3
%input.row = mul i32 %row, %input.row.stride
%output.row = mul i32 %row, %from
br label %attention.matrix.stage.channel.loop
attention.matrix.stage.channel.loop:
%stage.channel.local = phi i32 [ %lid, %attention.matrix.job.step ], [ %stage.channel.next, %attention.matrix.stage.channel.done ]
%stage.channel.more = icmp ult i32 %stage.channel.local, %head.width
br i1 %stage.channel.more, label %attention.matrix.stage.channel.step, label %attention.matrix.stage.done
attention.matrix.stage.channel.step:
%stage.channel = add i32 %head.start, %stage.channel.local
%stage.channel.base = mul i32 %stage.channel, %length
br label %attention.matrix.stage.plane.loop
attention.matrix.stage.plane.loop:
%stage.plane = phi i32 [ 0, %attention.matrix.stage.channel.step ], [ %stage.plane.next, %attention.matrix.stage.plane.done ]
%stage.plane.more = icmp ult i32 %stage.plane, 4
br i1 %stage.plane.more, label %attention.matrix.stage.plane.step, label %attention.matrix.stage.channel.done
attention.matrix.stage.plane.step:
%stage.input.plane = mul i32 %stage.plane, %from
%stage.input.row = add i32 %input.row, %stage.input.plane
%stage.input.base = add i32 %stage.input.row, %stage.channel.base
%stage.delta.base = add i32 %output.row, %stage.channel.base
%stage.shared.base = mul i32 %stage.plane, %head.values
%stage.is.delta = icmp eq i32 %stage.plane, 3
br label %attention.matrix.stage.position.loop
attention.matrix.stage.position.loop:
%stage.position = phi i32 [ 0, %attention.matrix.stage.plane.step ], [ %stage.position.next, %attention.matrix.stage.position.advance ]
%stage.position.more = icmp ult i32 %stage.position, %length
br i1 %stage.position.more, label %attention.matrix.stage.vector.test, label %attention.matrix.stage.plane.done
attention.matrix.stage.vector.test:
%stage.position.remaining = sub i32 %length, %stage.position
%stage.vector = icmp uge i32 %stage.position.remaining, 16
br i1 %stage.vector, label %attention.matrix.stage.vector.select, label %attention.matrix.stage.scalar.select
attention.matrix.stage.vector.select:
br i1 %stage.is.delta, label %attention.matrix.stage.vector.delta, label %attention.matrix.stage.vector.input
attention.matrix.stage.vector.input:
%stage.vector.input.index = add i32 %stage.input.base, %stage.position
%stage.vector.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %stage.vector.input.index
%stage.vector.input.value = load <16 x double>, ptr addrspace(1) %stage.vector.input.ptr, align 8
br label %attention.matrix.stage.vector.store
attention.matrix.stage.vector.delta:
%stage.vector.delta.index = add i32 %stage.delta.base, %stage.position
%stage.vector.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %stage.vector.delta.index
%stage.vector.delta.value = load <16 x double>, ptr addrspace(1) %stage.vector.delta.ptr, align 8
br label %attention.matrix.stage.vector.store
attention.matrix.stage.vector.store:
%stage.vector.value = phi <16 x double> [ %stage.vector.input.value, %attention.matrix.stage.vector.input ], [ %stage.vector.delta.value, %attention.matrix.stage.vector.delta ]
call void @contraction_stage_column16(<16 x double> %stage.vector.value, i32 %stage.shared.base, i32 %stage.position, i32 %stage.channel.local, i32 %head.width)
%stage.vector.position.next = add i32 %stage.position, 16
br label %attention.matrix.stage.position.advance
attention.matrix.stage.scalar.select:
br i1 %stage.is.delta, label %attention.matrix.stage.scalar.delta, label %attention.matrix.stage.scalar.input
attention.matrix.stage.scalar.input:
%stage.scalar.input.index = add i32 %stage.input.base, %stage.position
%stage.scalar.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %stage.scalar.input.index
%stage.scalar.input.value = load double, ptr addrspace(1) %stage.scalar.input.ptr, align 8
br label %attention.matrix.stage.scalar.store
attention.matrix.stage.scalar.delta:
%stage.scalar.delta.index = add i32 %stage.delta.base, %stage.position
%stage.scalar.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %stage.scalar.delta.index
%stage.scalar.delta.value = load double, ptr addrspace(1) %stage.scalar.delta.ptr, align 8
br label %attention.matrix.stage.scalar.store
attention.matrix.stage.scalar.store:
%stage.scalar.value = phi double [ %stage.scalar.input.value, %attention.matrix.stage.scalar.input ], [ %stage.scalar.delta.value, %attention.matrix.stage.scalar.delta ]
%stage.scalar.row = mul i32 %stage.position, %head.width
%stage.scalar.local = add i32 %stage.scalar.row, %stage.channel.local
%stage.scalar.index.shared = add i32 %stage.shared.base, %stage.scalar.local
%stage.scalar.ptr.shared = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %stage.scalar.index.shared
store double %stage.scalar.value, ptr addrspace(3) %stage.scalar.ptr.shared, align 8
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
call void @recipe.local.barrier()
br label %attention.matrix.d.loop
attention.matrix.d.loop:
%d.query = phi i32 [ %lid, %attention.matrix.stage.done ], [ %d.query.next, %attention.matrix.d.store ]
%d.more = icmp ult i32 %d.query, %length
br i1 %d.more, label %attention.matrix.d.sum.loop, label %attention.matrix.d.done
attention.matrix.d.sum.loop:
%d.channel = phi i32 [ 0, %attention.matrix.d.loop ], [ %d.channel.next, %attention.matrix.d.sum.step ]
%d.sum = phi double [ 0.0, %attention.matrix.d.loop ], [ %d.sum.next, %attention.matrix.d.sum.step ]
%d.channel.more = icmp ult i32 %d.channel, %head.width
br i1 %d.channel.more, label %attention.matrix.d.sum.step, label %attention.matrix.d.store
attention.matrix.d.sum.step:
%d.shared.row = mul i32 %d.query, %head.width
%d.shared.local = add i32 %d.shared.row, %d.channel
%d.do.index = add i32 %do.base, %d.shared.local
%d.do.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %d.do.index
%d.do = load double, ptr addrspace(3) %d.do.ptr, align 8
%d.output.channel = add i32 %head.start, %d.channel
%d.output.channel.base = mul i32 %d.output.channel, %length
%d.output.local = add i32 %d.output.channel.base, %d.query
%d.output.index = add i32 %output.row, %d.output.local
%d.output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %d.output.index
%d.output = load double, ptr addrspace(1) %d.output.ptr, align 8
%d.term = call double @recipe.mul(double %d.do, double %d.output)
%d.sum.next = call double @recipe.add(double %d.sum, double %d.term)
%d.channel.next = add i32 %d.channel, 1
br label %attention.matrix.d.sum.loop
attention.matrix.d.store:
%d.index = add i32 %d.base, %d.query
%d.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %d.index
store double %d.sum, ptr addrspace(3) %d.ptr, align 8
%d.query.next = add i32 %d.query, %block
br label %attention.matrix.d.loop
attention.matrix.d.done:
call void @recipe.local.barrier()
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
%matrix.score.accumulator = phi <8 x RECIPE_STATE> [ zeroinitializer, %attention.matrix.score.job.step ], [ %matrix.score.accumulator.next, %attention.matrix.score.width.step ]
%matrix.dp.accumulator = phi <8 x RECIPE_STATE> [ zeroinitializer, %attention.matrix.score.job.step ], [ %matrix.dp.accumulator.next, %attention.matrix.score.width.step ]
%matrix.width.more = icmp ult i32 %matrix.width, %head.width
br i1 %matrix.width.more, label %attention.matrix.score.width.step, label %attention.matrix.score.store.loop
attention.matrix.score.width.step:
%matrix.q.row = mul i32 %matrix.q.safe, %head.width
%matrix.q.local = add i32 %matrix.q.row, %matrix.width
%matrix.q.index = add i32 %q.base, %matrix.q.local
%matrix.q.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.q.index
%matrix.q.fragment = load <16 x double>, ptr addrspace(3) %matrix.q.ptr, align 2
%matrix.k.row = mul i32 %matrix.k.safe, %head.width
%matrix.k.local = add i32 %matrix.k.row, %matrix.width
%matrix.k.index = add i32 %k.base, %matrix.k.local
%matrix.k.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.k.index
%matrix.k.fragment = load <16 x double>, ptr addrspace(3) %matrix.k.ptr, align 2
%matrix.do.index = add i32 %do.base, %matrix.q.local
%matrix.do.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.do.index
%matrix.do.fragment = load <16 x double>, ptr addrspace(3) %matrix.do.ptr, align 2
%matrix.v.index = add i32 %v.base, %matrix.k.local
%matrix.v.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.v.index
%matrix.v.fragment = load <16 x double>, ptr addrspace(3) %matrix.v.ptr, align 2
%matrix.score.accumulator.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %matrix.q.fragment, <16 x double> %matrix.k.fragment, <8 x RECIPE_STATE> %matrix.score.accumulator)
%matrix.dp.accumulator.next = call <8 x RECIPE_STATE> @recipe.wmma(<16 x double> %matrix.do.fragment, <16 x double> %matrix.v.fragment, <8 x RECIPE_STATE> %matrix.dp.accumulator)
%matrix.width.next = add i32 %matrix.width, 16
br label %attention.matrix.score.width.loop
attention.matrix.score.store.loop:
%matrix.output = phi i32 [ 0, %attention.matrix.score.width.loop ], [ %matrix.output.next, %attention.matrix.score.store.next ]
%matrix.output.more = icmp ult i32 %matrix.output, 8
br i1 %matrix.output.more, label %attention.matrix.score.store.test, label %attention.matrix.score.store.done
attention.matrix.score.store.test:
%matrix.output.twice = mul i32 %matrix.output, 2
%matrix.query.local = add i32 %matrix.output.twice, %matrix.lane.half
%matrix.query = add i32 %matrix.q.tile, %matrix.query.local
%matrix.query.valid = icmp ult i32 %matrix.query, %length
%matrix.pair.valid = and i1 %matrix.query.valid, %matrix.k.valid
br i1 %matrix.pair.valid, label %attention.matrix.score.complete, label %attention.matrix.score.store.next
attention.matrix.score.complete:
%matrix.score.wide = extractelement <8 x RECIPE_STATE> %matrix.score.accumulator, i32 %matrix.output
%matrix.score.raw = call double @recipe.encode(RECIPE_STATE %matrix.score.wide)
%matrix.score = call double @recipe.div(double %matrix.score.raw, double %scale)
%matrix.dp.wide = extractelement <8 x RECIPE_STATE> %matrix.dp.accumulator, i32 %matrix.output
%matrix.dp = call double @recipe.encode(RECIPE_STATE %matrix.dp.wide)
%matrix.causal = icmp ule i32 %matrix.k, %matrix.query
%matrix.statistics.base = mul i32 %head.job, %length
%matrix.statistics.index = add i32 %matrix.statistics.base, %matrix.query
%matrix.maximum.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %matrix.statistics.index
%matrix.maximum = load double, ptr addrspace(1) %matrix.maximum.ptr, align 8
%matrix.denominator.index = add i32 %statistics.rows, %matrix.statistics.index
%matrix.denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %matrix.denominator.index
%matrix.denominator = load double, ptr addrspace(1) %matrix.denominator.ptr, align 8
%matrix.centered = call double @recipe.sub(double %matrix.score, double %matrix.maximum)
%matrix.exponential = call double @recipe.exp(double %matrix.centered)
%matrix.probability.raw = call double @recipe.div(double %matrix.exponential, double %matrix.denominator)
%matrix.probability = select i1 %matrix.causal, double %matrix.probability.raw, double 0.0
%matrix.d.index = add i32 %d.base, %matrix.query
%matrix.d.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.d.index
%matrix.d = load double, ptr addrspace(3) %matrix.d.ptr, align 8
%matrix.dp.centered = call double @recipe.sub(double %matrix.dp, double %matrix.d)
%matrix.derivative = call double @recipe.mul(double %matrix.probability, double %matrix.dp.centered)
%matrix.pair.row = mul i32 %matrix.query, %length
%matrix.pair.local = add i32 %matrix.pair.row, %matrix.k
%matrix.p.index = add i32 %p.base, %matrix.pair.local
%matrix.p.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.p.index
store double %matrix.probability, ptr addrspace(3) %matrix.p.ptr, align 8
%matrix.ds.index = add i32 %ds.base, %matrix.pair.local
%matrix.ds.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %matrix.ds.index
store double %matrix.derivative, ptr addrspace(3) %matrix.ds.ptr, align 8
br label %attention.matrix.score.store.next
attention.matrix.score.store.next:
%matrix.output.next = add i32 %matrix.output, 1
br label %attention.matrix.score.store.loop
attention.matrix.score.store.done:
%matrix.job.next = add i32 %matrix.job, %matrix.waves
br label %attention.matrix.score.job.loop
attention.matrix.score.done:
call void @recipe.local.barrier()
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 0, i32 %ds.base, i32 %k.base, i32 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 1, i32 %ds.base, i32 %q.base, i32 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 2, i32 %p.base, i32 %do.base, i32 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
call void @recipe.local.barrier()
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
i32 %kv.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%head.width.double = call double @recipe.from.u32(i32 %head.width)
%scale = call double @recipe.sqrt(double %head.width.double)
%kv.group = udiv i32 %heads, %kv.heads
%kv.channels = mul i32 %kv.heads, %head.width
%kv.plane = mul i32 %kv.channels, %length
%kv.planes = mul i32 %kv.plane, 2
%value.plane.base = add i32 %from, %kv.plane
%index.query.channels = mul i32 %index.heads, %index.width
%index.channels = add i32 %index.query.channels, %index.width
%index.plane = mul i32 %index.channels, %length
%gate.plane = select i1 %gate, i32 %from, i32 0
%index.query.base = add i32 %from, %kv.planes
%gate.base = add i32 %index.query.base, %index.plane
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
%score.total = mul i32 %score.row.stride, %rows
%derivative.base = add i32 %score.base, %score.total
%derivative.head.stride = mul i32 %length, %blocks
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
%dq.row = udiv i32 %dq.head.job, %heads
%dq.query.base = mul i32 %dq.query.tile, %tile.m
%dq.query.remaining = sub i32 %length, %dq.query.base
%dq.query.short = icmp ult i32 %dq.query.remaining, %tile.m
%dq.query.count = select i1 %dq.query.short, i32 %dq.query.remaining, i32 %tile.m
%dq.query.last = add i32 %dq.query.base, %dq.query.count
%dq.head.start = mul i32 %dq.head, %head.width
%dq.kv.head = udiv i32 %dq.head, %kv.group
%dq.kv.head.start = mul i32 %dq.kv.head, %head.width
%dq.input.row = mul i32 %dq.row, %row.stride
%dq.output.row = mul i32 %dq.row, %from
%dq.score.row = mul i32 %dq.row, %score.row.stride
%dq.score.row.base = add i32 %score.base, %dq.score.row
%dq.active.query.values = mul i32 %dq.query.count, %head.width
br label %dq.query.stage.loop
dq.query.stage.loop:
%dq.query.p = phi i32 [ %lid, %dq.job.prepare ], [ %dq.query.p.next, %dq.query.stage.step ]
%dq.query.p.more = icmp ult i32 %dq.query.p, %dq.active.query.values
br i1 %dq.query.p.more, label %dq.query.stage.step, label %dq.query.stage.done
dq.query.stage.step:
%dq.query.local = udiv i32 %dq.query.p, %head.width
%dq.channel.local = urem i32 %dq.query.p, %head.width
%dq.query.position = add i32 %dq.query.base, %dq.query.local
%dq.channel = add i32 %dq.head.start, %dq.channel.local
%dq.channel.base = mul i32 %dq.channel, %length
%dq.local = add i32 %dq.channel.base, %dq.query.position
%dq.input.index = add i32 %dq.input.row, %dq.local
%dq.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dq.input.index
%dq.query.value = load double, ptr addrspace(1) %dq.input.ptr, align 8
%dq.query.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.query.p
store double %dq.query.value, ptr addrspace(3) %dq.query.shared.ptr, align 8
%dq.delta.index = add i32 %dq.output.row, %dq.local
%dq.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %dq.delta.index
%dq.delta.value = load double, ptr addrspace(1) %dq.delta.ptr, align 8
%dq.delta.shared.index = add i32 %dq.delta.base.shared, %dq.query.p
%dq.delta.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.delta.shared.index
store double %dq.delta.value, ptr addrspace(3) %dq.delta.shared.ptr, align 8
%dq.gradient.shared.index = add i32 %dq.gradient.base.shared, %dq.query.p
%dq.gradient.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.gradient.shared.index
store double 0.0, ptr addrspace(3) %dq.gradient.shared.ptr, align 8
%dq.query.p.next = add i32 %dq.query.p, %block
br label %dq.query.stage.loop
dq.query.stage.done:
call void @recipe.local.barrier()
call void @attention_tile_products(ptr addrspace(1) %output, i32 %dq.output.row, i32 %dq.delta.base.shared,
i32 %dq.product.base.shared, i32 %dq.query.base, i32 %dq.query.count, i32 %dq.head.start,
i32 %head.width, i32 %length, i32 %lid, i32 %block)
call void @recipe.local.barrier()
br label %dq.norm.done
dq.norm.done:
br i1 %gate, label %dq.gate.loop, label %dq.gate.done
dq.gate.loop:
%dq.gate.p = phi i32 [ %lid, %dq.norm.done ], [ %dq.gate.p.next, %dq.gate.step ]
%dq.gate.more = icmp ult i32 %dq.gate.p, %dq.active.query.values
br i1 %dq.gate.more, label %dq.gate.step, label %dq.gate.exit
dq.gate.step:
%dq.gate.query = udiv i32 %dq.gate.p, %head.width
%dq.gate.channel = urem i32 %dq.gate.p, %head.width
%dq.gate.position = add i32 %dq.query.base, %dq.gate.query
%dq.gate.output.channel = add i32 %dq.head.start, %dq.gate.channel
%dq.gate.channel.base = mul i32 %dq.gate.output.channel, %length
%dq.gate.local = add i32 %dq.gate.channel.base, %dq.gate.position
%dq.gate.row = add i32 %dq.input.row, %gate.base
%dq.gate.index = add i32 %dq.gate.row, %dq.gate.local
%dq.gate.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dq.gate.index
%dq.gate.value = load double, ptr addrspace(1) %dq.gate.ptr, align 8
%dq.gate.factor = call double @recipe.sigmoid(double %dq.gate.value)
%dq.gate.shared.index = add i32 %dq.delta.base.shared, %dq.gate.p
%dq.gate.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.gate.shared.index
%dq.gate.delta = load double, ptr addrspace(3) %dq.gate.shared.ptr, align 8
%dq.gate.output.index = add i32 %dq.output.row, %dq.gate.local
%dq.gate.output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %dq.gate.output.index
%dq.gate.output.value = load double, ptr addrspace(1) %dq.gate.output.ptr, align 8
%dq.gate.one = call double @recipe.from.u1(i1 true)
%dq.gate.complement = call double @recipe.sub(double %dq.gate.one, double %dq.gate.factor)
%dq.gate.product = call double @recipe.mul(double %dq.gate.delta, double %dq.gate.output.value)
%dq.gate.gradient = call double @recipe.mul(double %dq.gate.product, double %dq.gate.complement)
%dq.gate.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %dq.gate.index
store double %dq.gate.gradient, ptr addrspace(1) %dq.gate.previous.ptr, align 8
%dq.gate.scaled = call double @recipe.mul(double %dq.gate.delta, double %dq.gate.factor)
store double %dq.gate.scaled, ptr addrspace(3) %dq.gate.shared.ptr, align 8
%dq.gate.p.next = add i32 %dq.gate.p, %block
br label %dq.gate.loop
dq.gate.exit:
call void @recipe.local.barrier()
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
%dq.scan.kept = call i1 @attention_selected(ptr addrspace(1) %context, i32 %dq.score.row.base, i32 %blocks, i32 %select.block, i32 %dq.scan.query.index, i32 %dq.scan.key)
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
%dq.key.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dq.key.input.index
%dq.key.value = load double, ptr addrspace(1) %dq.key.input.ptr, align 8
%dq.key.shared.index = add i32 %dq.key.base.shared, %dq.key.p
%dq.key.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.key.shared.index
store double %dq.key.value, ptr addrspace(3) %dq.key.shared.ptr, align 8
%dq.value.row = add i32 %dq.input.row, %value.plane.base
%dq.value.input.index = add i32 %dq.value.row, %dq.key.input.local
%dq.value.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dq.value.input.index
%dq.value.value = load double, ptr addrspace(1) %dq.value.input.ptr, align 8
%dq.value.shared.index = add i32 %dq.value.base.shared, %dq.key.p
%dq.value.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.value.shared.index
store double %dq.value.value, ptr addrspace(3) %dq.value.shared.ptr, align 8
%dq.key.p.next = add i32 %dq.key.p, %block
br label %dq.key.stage.loop
dq.key.stage.done:
call void @recipe.local.barrier()
br label %dq.key.norm.done
dq.key.norm.done:
call void @attention_tile_derivatives(ptr addrspace(1) %context, i32 0, i32 %dq.key.base.shared,
i32 %dq.delta.base.shared, i32 %dq.value.base.shared, i32 %dq.probability.base.shared,
i32 %dq.derivative.base.shared, i32 %dq.product.base.shared, i32 %dq.query.base,
i32 %dq.key.tile.base, i32 %dq.query.count, i32 %dq.key.count, i32 %tile.n,
i32 %dq.head.job, i32 %length, i32 %statistics.denominator.base, i32 %head.width,
double %scale, i32 %lid, i32 %block, i32 %dq.score.row.base, i32 %blocks, i32 %select.block, i1 %select)
call void @recipe.local.barrier()
br i1 %select, label %dq.index.loop, label %dq.accumulate.loop
dq.index.loop:
%dq.index.p = phi i32 [ %lid, %dq.key.norm.done ], [ %dq.index.p.next, %dq.index.query.done ]
%dq.index.more = icmp ult i32 %dq.index.p, %dq.query.count
br i1 %dq.index.more, label %dq.index.prepare, label %dq.index.done
dq.index.prepare:
%dq.index.query = add i32 %dq.query.base, %dq.index.p
%dq.index.head.plane = mul i32 %dq.head.job, %derivative.head.stride
%dq.index.head.base = add i32 %derivative.base, %dq.index.head.plane
%dq.index.query.offset = mul i32 %dq.index.query, %blocks
%dq.index.start = add i32 %dq.index.head.base, %dq.index.query.offset
%dq.index.pair.row = mul i32 %dq.index.p, %tile.n
br label %dq.index.key.loop
dq.index.key.loop:
%dq.index.key = phi i32 [ 0, %dq.index.prepare ], [ %dq.index.key.next, %dq.index.key.step ]
%dq.index.key.more = icmp ult i32 %dq.index.key, %dq.key.count
br i1 %dq.index.key.more, label %dq.index.key.step, label %dq.index.query.done
dq.index.key.step:
%dq.index.pair.local = add i32 %dq.index.pair.row, %dq.index.key
%dq.index.pair.index = add i32 %dq.derivative.base.shared, %dq.index.pair.local
%dq.index.pair.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.index.pair.index
%dq.index.derivative = load double, ptr addrspace(3) %dq.index.pair.ptr, align 8
%dq.index.key.position = add i32 %dq.key.tile.base, %dq.index.key
%dq.index.block = udiv i32 %dq.index.key.position, %select.block
%dq.index.slot = add i32 %dq.index.start, %dq.index.block
%dq.index.slot.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %dq.index.slot
%dq.index.prior = load double, ptr addrspace(1) %dq.index.slot.ptr, align 8
%dq.index.sum = call double @recipe.add(double %dq.index.prior, double %dq.index.derivative)
store double %dq.index.sum, ptr addrspace(1) %dq.index.slot.ptr, align 8
%dq.index.key.next = add i32 %dq.index.key, 1
br label %dq.index.key.loop
dq.index.query.done:
%dq.index.p.next = add i32 %dq.index.p, %block
br label %dq.index.loop
dq.index.done:
br label %dq.accumulate.loop
dq.accumulate.loop:
%dq.accumulate.p = phi i32 [ %lid, %dq.key.norm.done ], [ %lid, %dq.index.done ], [ %dq.accumulate.p.next, %dq.accumulate.store ]
%dq.accumulate.p.more = icmp ult i32 %dq.accumulate.p, %dq.active.query.values
br i1 %dq.accumulate.p.more, label %dq.accumulate.prepare, label %dq.accumulate.done
dq.accumulate.prepare:
%dq.accumulate.query = udiv i32 %dq.accumulate.p, %head.width
%dq.accumulate.channel = urem i32 %dq.accumulate.p, %head.width
%dq.accumulate.gradient.index = add i32 %dq.gradient.base.shared, %dq.accumulate.p
%dq.accumulate.gradient.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.accumulate.gradient.index
%dq.accumulate.initial = load double, ptr addrspace(3) %dq.accumulate.gradient.ptr, align 8
br label %dq.accumulate.key.loop
dq.accumulate.key.loop:
%dq.accumulate.key = phi i32 [ 0, %dq.accumulate.prepare ], [ %dq.accumulate.key.next, %dq.accumulate.key.step ]
%dq.accumulate.value = phi double [ %dq.accumulate.initial, %dq.accumulate.prepare ], [ %dq.accumulate.next, %dq.accumulate.key.step ]
%dq.accumulate.key.more = icmp ult i32 %dq.accumulate.key, %dq.key.count
br i1 %dq.accumulate.key.more, label %dq.accumulate.key.step, label %dq.accumulate.store
dq.accumulate.key.step:
%dq.accumulate.pair.row = mul i32 %dq.accumulate.query, %tile.n
%dq.accumulate.pair.local = add i32 %dq.accumulate.pair.row, %dq.accumulate.key
%dq.accumulate.pair.index = add i32 %dq.derivative.base.shared, %dq.accumulate.pair.local
%dq.accumulate.pair.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.accumulate.pair.index
%dq.accumulate.ds = load double, ptr addrspace(3) %dq.accumulate.pair.ptr, align 8
%dq.accumulate.key.row = mul i32 %dq.accumulate.key, %head.width
%dq.accumulate.key.local = add i32 %dq.accumulate.key.row, %dq.accumulate.channel
%dq.accumulate.key.index = add i32 %dq.key.base.shared, %dq.accumulate.key.local
%dq.accumulate.key.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.accumulate.key.index
%dq.accumulate.key.value = load double, ptr addrspace(3) %dq.accumulate.key.ptr, align 8
%dq.accumulate.raw = call double @recipe.mul(double %dq.accumulate.ds, double %dq.accumulate.key.value)
%dq.accumulate.term = call double @recipe.div(double %dq.accumulate.raw, double %scale)
%dq.accumulate.next = call double @recipe.add(double %dq.accumulate.value, double %dq.accumulate.term)
%dq.accumulate.key.next = add i32 %dq.accumulate.key, 1
br label %dq.accumulate.key.loop
dq.accumulate.store:
store double %dq.accumulate.value, ptr addrspace(3) %dq.accumulate.gradient.ptr, align 8
%dq.accumulate.p.next = add i32 %dq.accumulate.p, %block
br label %dq.accumulate.loop
dq.accumulate.done:
call void @recipe.local.barrier()
br label %dq.key.tile.advance
dq.key.tile.advance:
%dq.key.tile.next = add i32 %dq.key.tile.base, %tile.n
br label %dq.key.tile.loop
dq.store.begin:
call void @recipe.local.barrier()
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
%dq.store.ptr = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %dq.store.index
%dq.store.shared.index = add i32 %dq.gradient.base.shared, %dq.store.p
%dq.store.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dq.store.shared.index
%dq.store.value = load double, ptr addrspace(3) %dq.store.shared.ptr, align 8
store double %dq.store.value, ptr addrspace(1) %dq.store.ptr, align 8
%dq.store.p.next = add i32 %dq.store.p, %block
br label %dq.store.loop
dq.store.done:
call void @recipe.local.barrier()
br label %dq.job.finish
dq.job.finish:
%dq.job.next = add i32 %dq.job, %groups
br label %dq.job.loop
dq.exit:
%dkv.head.jobs = mul i32 %rows, %kv.heads
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
%dkv.kv.head = urem i32 %dkv.kv.job, %kv.heads
%dkv.row = udiv i32 %dkv.kv.job, %kv.heads
%dkv.key.base = mul i32 %dkv.key.tile, %tile.n
%dkv.key.remaining = sub i32 %length, %dkv.key.base
%dkv.key.short = icmp ult i32 %dkv.key.remaining, %tile.n
%dkv.key.count = select i1 %dkv.key.short, i32 %dkv.key.remaining, i32 %tile.n
%dkv.kv.head.start = mul i32 %dkv.kv.head, %head.width
%dkv.input.row = mul i32 %dkv.row, %row.stride
%dkv.output.row = mul i32 %dkv.row, %from
%dkv.score.row = mul i32 %dkv.row, %score.row.stride
%dkv.score.row.base = add i32 %score.base, %dkv.score.row
%dkv.active.key.values = mul i32 %dkv.key.count, %head.width
%dkv.head.row = mul i32 %dkv.row, %heads
%dkv.head.base = mul i32 %dkv.kv.head, %kv.group
br label %dkv.key.stage.loop
dkv.key.stage.loop:
%dkv.key.p = phi i32 [ %lid, %dkv.job.prepare ], [ %dkv.key.p.next, %dkv.key.stage.step ]
%dkv.key.p.more = icmp ult i32 %dkv.key.p, %dkv.active.key.values
br i1 %dkv.key.p.more, label %dkv.key.stage.step, label %dkv.key.stage.done
dkv.key.stage.step:
%dkv.key.local = udiv i32 %dkv.key.p, %head.width
%dkv.channel.local = urem i32 %dkv.key.p, %head.width
%dkv.key.position = add i32 %dkv.key.base, %dkv.key.local
%dkv.channel = add i32 %dkv.kv.head.start, %dkv.channel.local
%dkv.channel.base = mul i32 %dkv.channel, %length
%dkv.local = add i32 %dkv.channel.base, %dkv.key.position
%dkv.key.plane = add i32 %dkv.input.row, %from
%dkv.key.input.index = add i32 %dkv.key.plane, %dkv.local
%dkv.key.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dkv.key.input.index
%dkv.key.value = load double, ptr addrspace(1) %dkv.key.input.ptr, align 8
%dkv.key.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.key.p
store double %dkv.key.value, ptr addrspace(3) %dkv.key.shared.ptr, align 8
%dkv.value.row = add i32 %dkv.input.row, %value.plane.base
%dkv.value.input.index = add i32 %dkv.value.row, %dkv.local
%dkv.value.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dkv.value.input.index
%dkv.value.value = load double, ptr addrspace(1) %dkv.value.input.ptr, align 8
%dkv.value.shared.index = add i32 %dkv.value.base.shared, %dkv.key.p
%dkv.value.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.value.shared.index
store double %dkv.value.value, ptr addrspace(3) %dkv.value.shared.ptr, align 8
%dkv.key.gradient.index = add i32 %dkv.key.gradient.base.shared, %dkv.key.p
%dkv.key.gradient.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.key.gradient.index
store double 0.0, ptr addrspace(3) %dkv.key.gradient.ptr, align 8
%dkv.value.gradient.index = add i32 %dkv.value.gradient.base.shared, %dkv.key.p
%dkv.value.gradient.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.value.gradient.index
store double 0.0, ptr addrspace(3) %dkv.value.gradient.ptr, align 8
%dkv.key.p.next = add i32 %dkv.key.p, %block
br label %dkv.key.stage.loop
dkv.key.stage.done:
call void @recipe.local.barrier()
br label %dkv.key.norm.done
dkv.key.norm.done:
br label %dkv.head.loop
dkv.head.loop:
%dkv.head.slot = phi i32 [ 0, %dkv.key.norm.done ], [ %dkv.head.slot.next, %dkv.head.step ]
%dkv.head.more = icmp ult i32 %dkv.head.slot, %kv.group
br i1 %dkv.head.more, label %dkv.head.prepare, label %dkv.store.begin
dkv.head.prepare:
%dkv.head = add i32 %dkv.head.base, %dkv.head.slot
%dkv.head.start = mul i32 %dkv.head, %head.width
%dkv.head.job = add i32 %dkv.head.row, %dkv.head
br label %dkv.query.tile.loop
dkv.query.tile.loop:
%dkv.query.base = phi i32 [ %dkv.key.base, %dkv.head.prepare ], [ %dkv.query.next, %dkv.query.advance ]
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
%dkv.scan.kept = call i1 @attention_selected(ptr addrspace(1) %context, i32 %dkv.score.row.base, i32 %blocks, i32 %select.block, i32 %dkv.scan.query.index, i32 %dkv.scan.key)
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
%dkv.query.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dkv.query.input.index
%dkv.query.value = load double, ptr addrspace(1) %dkv.query.input.ptr, align 8
%dkv.query.shared.index = add i32 %dkv.query.base.shared, %dkv.query.p
%dkv.query.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.query.shared.index
store double %dkv.query.value, ptr addrspace(3) %dkv.query.shared.ptr, align 8
%dkv.delta.input.index = add i32 %dkv.output.row, %dkv.query.input.local
%dkv.delta.input.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %dkv.delta.input.index
%dkv.delta.value = load double, ptr addrspace(1) %dkv.delta.input.ptr, align 8
%dkv.delta.shared.index = add i32 %dkv.delta.base.shared, %dkv.query.p
%dkv.delta.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.delta.shared.index
store double %dkv.delta.value, ptr addrspace(3) %dkv.delta.shared.ptr, align 8
%dkv.query.p.next = add i32 %dkv.query.p, %block
br label %dkv.query.stage.loop
dkv.query.stage.done:
call void @recipe.local.barrier()
call void @attention_tile_products(ptr addrspace(1) %output, i32 %dkv.output.row, i32 %dkv.delta.base.shared,
i32 %dkv.product.base.shared, i32 %dkv.query.base, i32 %dkv.query.count, i32 %dkv.head.start,
i32 %head.width, i32 %length, i32 %lid, i32 %block)
call void @recipe.local.barrier()
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
%dkv.gate.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dkv.gate.index
%dkv.gate.value = load double, ptr addrspace(1) %dkv.gate.ptr, align 8
%dkv.gate.factor = call double @recipe.sigmoid(double %dkv.gate.value)
%dkv.gate.shared.index = add i32 %dkv.delta.base.shared, %dkv.gate.p
%dkv.gate.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.gate.shared.index
%dkv.gate.delta = load double, ptr addrspace(3) %dkv.gate.shared.ptr, align 8
%dkv.gate.scaled = call double @recipe.mul(double %dkv.gate.delta, double %dkv.gate.factor)
store double %dkv.gate.scaled, ptr addrspace(3) %dkv.gate.shared.ptr, align 8
%dkv.gate.p.next = add i32 %dkv.gate.p, %block
br label %dkv.gate.loop
dkv.gate.exit:
call void @recipe.local.barrier()
br label %dkv.gate.done
dkv.gate.done:
call void @attention_tile_derivatives(ptr addrspace(1) %context, i32 %dkv.query.base.shared, i32 0,
i32 %dkv.delta.base.shared, i32 %dkv.value.base.shared, i32 %dkv.probability.base.shared,
i32 %dkv.derivative.base.shared, i32 %dkv.product.base.shared, i32 %dkv.query.base,
i32 %dkv.key.base, i32 %dkv.query.count, i32 %dkv.key.count, i32 %tile.n,
i32 %dkv.head.job, i32 %length, i32 %statistics.denominator.base, i32 %head.width,
double %scale, i32 %lid, i32 %block, i32 %dkv.score.row.base, i32 %blocks, i32 %select.block, i1 %select)
call void @recipe.local.barrier()
br label %dkv.accumulate.loop
dkv.accumulate.loop:
%dkv.accumulate.p = phi i32 [ %lid, %dkv.gate.done ], [ %dkv.accumulate.p.next, %dkv.accumulate.store ]
%dkv.accumulate.p.more = icmp ult i32 %dkv.accumulate.p, %dkv.active.key.values
br i1 %dkv.accumulate.p.more, label %dkv.accumulate.prepare, label %dkv.accumulate.done
dkv.accumulate.prepare:
%dkv.accumulate.key = udiv i32 %dkv.accumulate.p, %head.width
%dkv.accumulate.channel = urem i32 %dkv.accumulate.p, %head.width
%dkv.accumulate.key.gradient.index = add i32 %dkv.key.gradient.base.shared, %dkv.accumulate.p
%dkv.accumulate.key.gradient.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.accumulate.key.gradient.index
%dkv.accumulate.key.initial = load double, ptr addrspace(3) %dkv.accumulate.key.gradient.ptr, align 8
%dkv.accumulate.value.gradient.index = add i32 %dkv.value.gradient.base.shared, %dkv.accumulate.p
%dkv.accumulate.value.gradient.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.accumulate.value.gradient.index
%dkv.accumulate.value.initial = load double, ptr addrspace(3) %dkv.accumulate.value.gradient.ptr, align 8
br label %dkv.accumulate.query.loop
dkv.accumulate.query.loop:
%dkv.accumulate.query = phi i32 [ 0, %dkv.accumulate.prepare ], [ %dkv.accumulate.query.next, %dkv.accumulate.query.step ]
%dkv.accumulate.key.value = phi double [ %dkv.accumulate.key.initial, %dkv.accumulate.prepare ], [ %dkv.accumulate.key.next, %dkv.accumulate.query.step ]
%dkv.accumulate.value.value = phi double [ %dkv.accumulate.value.initial, %dkv.accumulate.prepare ], [ %dkv.accumulate.value.next, %dkv.accumulate.query.step ]
%dkv.accumulate.query.more = icmp ult i32 %dkv.accumulate.query, %dkv.query.count
br i1 %dkv.accumulate.query.more, label %dkv.accumulate.query.step, label %dkv.accumulate.store
dkv.accumulate.query.step:
%dkv.accumulate.pair.row = mul i32 %dkv.accumulate.query, %tile.n
%dkv.accumulate.pair.local = add i32 %dkv.accumulate.pair.row, %dkv.accumulate.key
%dkv.accumulate.probability.index = add i32 %dkv.probability.base.shared, %dkv.accumulate.pair.local
%dkv.accumulate.probability.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.accumulate.probability.index
%dkv.accumulate.probability = load double, ptr addrspace(3) %dkv.accumulate.probability.ptr, align 8
%dkv.accumulate.derivative.index = add i32 %dkv.derivative.base.shared, %dkv.accumulate.pair.local
%dkv.accumulate.derivative.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.accumulate.derivative.index
%dkv.accumulate.derivative = load double, ptr addrspace(3) %dkv.accumulate.derivative.ptr, align 8
%dkv.accumulate.query.row = mul i32 %dkv.accumulate.query, %head.width
%dkv.accumulate.query.local = add i32 %dkv.accumulate.query.row, %dkv.accumulate.channel
%dkv.accumulate.query.index = add i32 %dkv.query.base.shared, %dkv.accumulate.query.local
%dkv.accumulate.query.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.accumulate.query.index
%dkv.accumulate.query.value = load double, ptr addrspace(3) %dkv.accumulate.query.ptr, align 8
%dkv.accumulate.delta.index = add i32 %dkv.delta.base.shared, %dkv.accumulate.query.local
%dkv.accumulate.delta.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.accumulate.delta.index
%dkv.accumulate.delta = load double, ptr addrspace(3) %dkv.accumulate.delta.ptr, align 8
%dkv.accumulate.key.raw = call double @recipe.mul(double %dkv.accumulate.derivative, double %dkv.accumulate.query.value)
%dkv.accumulate.key.term = call double @recipe.div(double %dkv.accumulate.key.raw, double %scale)
%dkv.accumulate.key.next = call double @recipe.add(double %dkv.accumulate.key.value, double %dkv.accumulate.key.term)
%dkv.accumulate.value.term = call double @recipe.mul(double %dkv.accumulate.probability, double %dkv.accumulate.delta)
%dkv.accumulate.value.next = call double @recipe.add(double %dkv.accumulate.value.value, double %dkv.accumulate.value.term)
%dkv.accumulate.query.next = add i32 %dkv.accumulate.query, 1
br label %dkv.accumulate.query.loop
dkv.accumulate.store:
store double %dkv.accumulate.key.value, ptr addrspace(3) %dkv.accumulate.key.gradient.ptr, align 8
store double %dkv.accumulate.value.value, ptr addrspace(3) %dkv.accumulate.value.gradient.ptr, align 8
%dkv.accumulate.p.next = add i32 %dkv.accumulate.p, %block
br label %dkv.accumulate.loop
dkv.accumulate.done:
call void @recipe.local.barrier()
br label %dkv.query.advance
dkv.query.advance:
%dkv.query.next = add i32 %dkv.query.base, %tile.m
br label %dkv.query.tile.loop
dkv.head.step:
%dkv.head.slot.next = add i32 %dkv.head.slot, 1
br label %dkv.head.loop
dkv.store.begin:
call void @recipe.local.barrier()
br label %dkv.adjoint.done
dkv.adjoint.done:
br label %dkv.store.loop
dkv.store.loop:
%dkv.store.p = phi i32 [ %lid, %dkv.adjoint.done ], [ %dkv.store.p.next, %dkv.store.step ]
%dkv.store.p.more = icmp ult i32 %dkv.store.p, %dkv.active.key.values
br i1 %dkv.store.p.more, label %dkv.store.step, label %dkv.store.done
dkv.store.step:
%dkv.store.key.local = udiv i32 %dkv.store.p, %head.width
%dkv.store.channel.local = urem i32 %dkv.store.p, %head.width
%dkv.store.key = add i32 %dkv.key.base, %dkv.store.key.local
%dkv.store.channel = add i32 %dkv.kv.head.start, %dkv.store.channel.local
%dkv.store.channel.base = mul i32 %dkv.store.channel, %length
%dkv.store.local = add i32 %dkv.store.channel.base, %dkv.store.key
%dkv.store.key.row = add i32 %dkv.input.row, %from
%dkv.store.key.index = add i32 %dkv.store.key.row, %dkv.store.local
%dkv.store.key.ptr = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %dkv.store.key.index
%dkv.store.key.shared.index = add i32 %dkv.key.gradient.base.shared, %dkv.store.p
%dkv.store.key.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.store.key.shared.index
%dkv.store.key.value = load double, ptr addrspace(3) %dkv.store.key.shared.ptr, align 8
store double %dkv.store.key.value, ptr addrspace(1) %dkv.store.key.ptr, align 8
%dkv.store.value.row = add i32 %dkv.input.row, %value.plane.base
%dkv.store.value.index = add i32 %dkv.store.value.row, %dkv.store.local
%dkv.store.value.ptr = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %dkv.store.value.index
%dkv.store.value.shared.index = add i32 %dkv.value.gradient.base.shared, %dkv.store.p
%dkv.store.value.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %dkv.store.value.shared.index
%dkv.store.value.value = load double, ptr addrspace(3) %dkv.store.value.shared.ptr, align 8
store double %dkv.store.value.value, ptr addrspace(1) %dkv.store.value.ptr, align 8
%dkv.store.p.next = add i32 %dkv.store.p, %block
br label %dkv.store.loop
dkv.store.done:
call void @recipe.local.barrier()
br label %dkv.job.finish
dkv.job.finish:
%dkv.job.next = add i32 %dkv.job, %groups
br label %dkv.job.loop
exit:
ret void
}
define internal void @scan_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output,
ptr addrspace(1) %context, i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels, i32 %gates,
i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads ) #3 { entry: %tid = call i32 @llvm.amdgcn.workitem.id.x()
%in.elements = mul i32 %in.channels, %length
%out.elements = mul i32 %out.channels, %length %input.matrix = mul i32 %in.channels, %out.channels
%state.matrix = mul i32 %out.channels, %out.channels %matrix.span = add i32 %input.matrix, %state.matrix
%gate.stride = add i32 %matrix.span, %out.channels %gate.batch = mul i32 %rows, %out.elements
br label %precompute.loop precompute.loop:
%precompute.gate = phi i32 [ 0, %entry ], [ %precompute.next, %precompute.step ]
%precompute.more = icmp ult i32 %precompute.gate, %gates
br i1 %precompute.more, label %precompute.step, label %precompute.done precompute.step:
%precompute.weight.offset = mul i32 %precompute.gate, %gate.stride
%precompute.weights = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %precompute.weight.offset
%precompute.context.offset = mul i32 %precompute.gate, %gate.batch
%precompute.context = getelementptr inbounds double, ptr addrspace(1) %context, i32 %precompute.context.offset
call void @contraction_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %precompute.weights,
ptr addrspace(1) %precompute.context, ptr addrspace(1) %input,
i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels,
i32 %length, i32 0, i1 false, i1 false, i1 false, i1 false, i1 false,
i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads )
%precompute.next = add i32 %precompute.gate, 1 br label %precompute.loop precompute.done:
call void @llvm.amdgcn.s.barrier() br label %row.loop row.loop:
%row = phi i32 [ %tid, %precompute.done ], [ %row.next, %time.done ] %row.more = icmp ult i32 %row, %rows
br i1 %row.more, label %time.loop, label %exit time.loop: %time = phi i32 [ 0, %row.loop ], [ %time.next, %output.done ]
%previous.exists = icmp ne i32 %time, 0 %output.row.base = mul i32 %row, %out.elements
%time.more = icmp ult i32 %time, %length br i1 %time.more, label %gate.loop, label %time.done gate.loop:
%gate = phi i32 [ 0, %time.loop ], [ %gate.next, %hidden.done ] %gate.more = icmp ult i32 %gate, %gates
br i1 %gate.more, label %hidden.loop, label %output.loop hidden.loop:
%hidden = phi i32 [ 0, %gate.loop ], [ %hidden.next, %gate.store ] %gate.weight.base = mul i32 %gate, %gate.stride
%hidden.more = icmp ult i32 %hidden, %out.channels br i1 %hidden.more, label %input.load, label %hidden.done
input.load: %input.gate.base = mul i32 %gate, %gate.batch %input.hidden.base = mul i32 %hidden, %length
%input.local = add i32 %input.hidden.base, %time %input.row.local = add i32 %output.row.base, %input.local
%input.index = add i32 %input.gate.base, %input.row.local
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %input.index
%input.sum = load double, ptr addrspace(1) %input.ptr, align 8 br label %state.sum.loop state.sum.loop:
%state.channel = phi i32 [ 0, %input.load ], [ %state.next, %state.sum.step ]
%state.sum = phi double [ %input.sum, %input.load ], [ %state.sum.next, %state.sum.step ]
%state.more = icmp ult i32 %state.channel, %out.channels br i1 %state.more, label %state.sum.step, label %gate.activate
state.sum.step: %previous.time = sub i32 %time, 1 %previous.safe = select i1 %previous.exists, i32 %previous.time, i32 0
%state.channel.base = mul i32 %state.channel, %length %previous.local = add i32 %state.channel.base, %previous.safe
%previous.index = add i32 %output.row.base, %previous.local
%previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %previous.index
%previous.loaded = load double, ptr addrspace(1) %previous.ptr, align 8
%previous = select i1 %previous.exists, double %previous.loaded, double 0.0 %candidate.gate = icmp eq i32 %gate, 2
%gru = icmp eq i32 %gates, 3 %reset.candidate = and i1 %gru, %candidate.gate
%reset.channel.base = mul i32 %state.channel, %length %reset.local = add i32 %reset.channel.base, %time
%reset.row.index = add i32 %output.row.base, %reset.local %reset.base = add i32 %gate.batch, %reset.row.index
%reset.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %reset.base
%reset = load double, ptr addrspace(1) %reset.ptr, align 8 %reset.state = call double @recipe.mul(double %reset, double %previous)
%state.value = select i1 %reset.candidate, double %reset.state, double %previous
%state.weight.base = add i32 %gate.weight.base, %input.matrix %state.weight.row = mul i32 %state.channel, %out.channels
%state.weight.local = add i32 %state.weight.row, %hidden
%state.weight.index = add i32 %state.weight.base, %state.weight.local
%state.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %state.weight.index
%state.weight = load double, ptr addrspace(1) %state.weight.ptr, align 8
%state.product = call double @recipe.mul(double %state.value, double %state.weight) %state.sum.next = call double @recipe.add(double %state.sum, double %state.product)
%state.next = add nuw i32 %state.channel, 1 br label %state.sum.loop gate.activate:
%bias.base = add i32 %gate.weight.base, %matrix.span %bias.index = add i32 %bias.base, %hidden
%bias.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %bias.index
%bias = load double, ptr addrspace(1) %bias.ptr, align 8 %linear = call double @recipe.add(double %state.sum, double %bias)
%rnn = icmp eq i32 %gates, 1 %last.gate = sub i32 %gates, 1 %candidate = icmp eq i32 %gate, %last.gate
%use.tanh = or i1 %rnn, %candidate %tanh.value = call double @recipe.tanh(double %linear)
%sigmoid.value = call double @sigmoid(double %linear)
%gate.value = select i1 %use.tanh, double %tanh.value, double %sigmoid.value br label %gate.store gate.store:
%gate.context.base = mul i32 %gate, %gate.batch %gate.hidden.base = mul i32 %hidden, %length
%gate.local = add i32 %gate.hidden.base, %time %gate.row.local = add i32 %output.row.base, %gate.local
%gate.index = add i32 %gate.context.base, %gate.row.local
%gate.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gate.index
store double %gate.value, ptr addrspace(1) %gate.ptr, align 8 %hidden.next = add nuw i32 %hidden, 1
br label %hidden.loop hidden.done: %gate.next = add nuw i32 %gate, 1 br label %gate.loop output.loop:
%output.hidden = phi i32 [ 0, %gate.loop ], [ %output.next, %output.store ]
%output.more = icmp ult i32 %output.hidden, %out.channels br i1 %output.more, label %output.step, label %output.done
output.step: %output.hidden.base = mul i32 %output.hidden, %length %output.local = add i32 %output.hidden.base, %time
%output.index = add i32 %output.row.base, %output.local
%gate0.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %output.index
%gate0 = load double, ptr addrspace(1) %gate0.ptr, align 8
%is.gru = icmp eq i32 %gates, 3 %is.lstm = icmp eq i32 %gates, 4
%gate1.raw = add i32 %gate.batch, %output.index %gate1.index = select i1 %is.lstm, i32 %gate1.raw, i32 %output.index
%gate1.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gate1.index
%gate1 = load double, ptr addrspace(1) %gate1.ptr, align 8 %gate2.base = mul i32 %gate.batch, 2
%gate2.raw = add i32 %gate2.base, %output.index %gate2.valid = or i1 %is.gru, %is.lstm
%gate2.index = select i1 %gate2.valid, i32 %gate2.raw, i32 %output.index
%gate2.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gate2.index
%gate2 = load double, ptr addrspace(1) %gate2.ptr, align 8 %gate3.base = mul i32 %gate.batch, 3
%gate3.raw = add i32 %gate3.base, %output.index %gate3.index = select i1 %is.lstm, i32 %gate3.raw, i32 %output.index
%gate3.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gate3.index
%gate3 = load double, ptr addrspace(1) %gate3.ptr, align 8 %output.previous.time = sub i32 %time, 1
%output.previous.safe = select i1 %previous.exists, i32 %output.previous.time, i32 0
%output.previous.local = add i32 %output.hidden.base, %output.previous.safe
%output.previous.index = add i32 %output.row.base, %output.previous.local
%output.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %output.previous.index
%output.previous.loaded = load double, ptr addrspace(1) %output.previous.ptr, align 8
%output.previous = select i1 %previous.exists, double %output.previous.loaded, double 0.0
%one.update = call double @recipe.sub(double 1.0, double %gate0) %gru.old = call double @recipe.mul(double %gate0, double %output.previous)
%gru.new = call double @recipe.mul(double %one.update, double %gate2) %gru.value = call double @recipe.add(double %gru.old, double %gru.new)
%cell.base = mul i32 %gate.batch, %gates %cell.index = add i32 %cell.base, %output.index
%cell.previous.index = add i32 %cell.base, %output.previous.index
%cell.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %cell.previous.index
%cell.previous.loaded = load double, ptr addrspace(1) %cell.previous.ptr, align 8
%cell.previous = select i1 %previous.exists, double %cell.previous.loaded, double 0.0
%cell.old = call double @recipe.mul(double %gate1, double %cell.previous) %cell.new = call double @recipe.mul(double %gate0, double %gate3)
%cell = call double @recipe.add(double %cell.old, double %cell.new)
%cell.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %cell.index
store double %cell, ptr addrspace(1) %cell.ptr, align 8 %cell.tanh = call double @recipe.tanh(double %cell)
%lstm.value = call double @recipe.mul(double %gate2, double %cell.tanh)
%rnn.or.gru = select i1 %is.gru, double %gru.value, double %gate0
%output.value = select i1 %is.lstm, double %lstm.value, double %rnn.or.gru br label %output.store output.store:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %output.index
store double %output.value, ptr addrspace(1) %output.ptr, align 8 %output.next = add nuw i32 %output.hidden, 1
br label %output.loop output.done: %time.next = add nuw i32 %time, 1 br label %time.loop time.done:
%row.next = add i32 %row, %threads br label %row.loop exit: ret void }
define internal void @contraction_reverse_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %delta, ptr addrspace(1) %previous, ptr addrspace(1) %gradient, i1 %write.input, i1 %has.bias, i1 %relu, i1 %matrix.gradient,
i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %kernel, i32 %offset,
i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) #1 { entry:
%sums = alloca [RECIPE_REGISTER_COUNT x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5)
%bias.sums = alloca [RECIPE_REGISTER_N x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5)
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %lid = call i32 @recipe.local.id.x() %group = call i32 @recipe.group.id.x() %block = call i32 @recipe.workgroup.size.x() %groups = udiv i32 %threads, %block
%in.elements = mul i32 %in.channels, %in.length %out.elements = mul i32 %out.channels, %out.length %is.conv = icmp ne i32 %kernel, 0
%span = select i1 %is.conv, i32 %kernel, i32 1 %window = mul i32 %in.channels, %span
%gradient.r.total = mul i32 %rows, %out.length
%gradient.matrix.values = mul i32 %out.channels, %window
%gradient.bias.values = select i1 %has.bias, i32 %out.channels, i32 0
%gradient.values = add i32 %gradient.matrix.values, %gradient.bias.values
; Split-K scratch rows are written by different workgroups. Pad each row so no
; two rows share a machine word and a partial store cannot lose a neighbour. The
; base is aligned by the host, so row zero starts on the same boundary.
%gradient.stride.raw = add i32 %gradient.values, RECIPE_SCRATCH_ROW_MASK
%gradient.stride = and i32 %gradient.stride.raw, RECIPE_SCRATCH_ROW_CLEAR
%gradient.scratch = getelementptr inbounds double, ptr addrspace(1) %gradient, i32 RECIPE_GRADIENT_SCRATCH_BASE
%gradient.m.short = icmp ult i32 %gradient.tile.m, %window %gradient.m.tile = select i1 %gradient.m.short, i32 %gradient.tile.m, i32 %window %gradient.n.short = icmp ult i32 %gradient.tile.n, %out.channels %gradient.n.tile = select i1 %gradient.n.short, i32 %gradient.tile.n, i32 %out.channels
%gradient.k.short = icmp ult i32 %gradient.tile.k, %gradient.r.total %gradient.k.tile = select i1 %gradient.k.short, i32 %gradient.tile.k, i32 %gradient.r.total
%gradient.m.adjusted = add i32 %window, %gradient.m.tile %gradient.m.numerator = sub i32 %gradient.m.adjusted, 1 %gradient.m.tiles = udiv i32 %gradient.m.numerator, %gradient.m.tile %gradient.n.adjusted = add i32 %out.channels, %gradient.n.tile %gradient.n.numerator = sub i32 %gradient.n.adjusted, 1 %gradient.n.tiles = udiv i32 %gradient.n.numerator, %gradient.n.tile
%gradient.jobs = mul i32 %gradient.m.tiles, %gradient.n.tiles
; The K extent is cut into one contiguous partition per RECIPE_CONTRACTION_SPLIT_SPAN
; elements, capped at RECIPE_CONTRACTION_K_PARTITIONS. The count and the
; boundaries are a function of the extent and two program constants. Neither the
; staged tile, the workgroup width, nor the number of compute units appears in
; the formula, so every backend sums the same partials and combines them in the
; same order, while a long K still spreads across enough workgroups to cover the
; device when the output produces few jobs.
%gradient.split.span = select i1 %matrix.gradient, i32 RECIPE_CONTRACTION_MATRIX_SPLIT_SPAN, i32 RECIPE_CONTRACTION_SPLIT_SPAN
%gradient.splits.adjusted = add i32 %gradient.r.total, %gradient.split.span
%gradient.splits.numerator = sub i32 %gradient.splits.adjusted, 1
%gradient.splits.raw = udiv i32 %gradient.splits.numerator, %gradient.split.span
%gradient.splits.large = icmp ugt i32 %gradient.splits.raw, RECIPE_CONTRACTION_K_PARTITIONS
%gradient.splits = select i1 %gradient.splits.large, i32 RECIPE_CONTRACTION_K_PARTITIONS, i32 %gradient.splits.raw
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
%gradient.m.group.short = icmp ult i32 %gradient.m.tiles, RECIPE_CONTRACTION_SWIZZLE_M %gradient.m.group.limit = select i1 %gradient.m.group.short, i32 %gradient.m.tiles, i32 RECIPE_CONTRACTION_SWIZZLE_M %gradient.group.width = mul i32 %gradient.m.group.limit, %gradient.n.tiles %gradient.group.index = udiv i32 %gradient.job, %gradient.group.width %gradient.m.group.base = mul i32 %gradient.group.index, %gradient.m.group.limit %gradient.m.group.remaining = sub i32 %gradient.m.tiles, %gradient.m.group.base %gradient.m.group.tail = icmp ult i32 %gradient.m.group.remaining, %gradient.m.group.limit %gradient.m.group.count = select i1 %gradient.m.group.tail, i32 %gradient.m.group.remaining, i32 %gradient.m.group.limit %gradient.group.local = urem i32 %gradient.job, %gradient.group.width %gradient.m.group.local = urem i32 %gradient.group.local, %gradient.m.group.count %gradient.m.index = add i32 %gradient.m.group.base, %gradient.m.group.local %gradient.n.index = udiv i32 %gradient.group.local, %gradient.m.group.count %gradient.m.base = mul i32 %gradient.m.index, %gradient.m.tile %gradient.n.base = mul i32 %gradient.n.index, %gradient.n.tile
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
%gradient.method.store = call i1 @contraction_store_lane(i1 %gradient.lane.store, i32 %lid)
%gradient.lane.n = udiv i32 %gradient.output.lane, %gradient.m.lanes
%gradient.lane.m = urem i32 %gradient.output.lane, %gradient.m.lanes
%gradient.output.m.base = mul i32 %gradient.lane.m, RECIPE_REGISTER_M %gradient.output.n.base = mul i32 %gradient.lane.n, RECIPE_REGISTER_N
%gradient.bias.first = icmp eq i32 %gradient.m.base, 0
%gradient.bias.channel = icmp ult i32 %lid, %gradient.n.count
%gradient.bias.owner = and i1 %gradient.bias.first, %gradient.bias.channel
%gradient.bias.enable = and i1 %has.bias, %gradient.bias.owner
br label %gradient.sum.init.loop gradient.sum.init.loop:
%gradient.sum.init = phi i32 [ 0, %gradient.job.step ], [ %gradient.sum.init.next, %gradient.sum.init.step ] %gradient.sum.init.more = icmp ult i32 %gradient.sum.init, RECIPE_REGISTER_COUNT br i1 %gradient.sum.init.more, label %gradient.sum.init.step, label %gradient.tile.loop
gradient.sum.init.step: %gradient.sum.init.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %gradient.sum.init store RECIPE_STATE %state.zero, ptr addrspace(5) %gradient.sum.init.ptr, align RECIPE_STATE_ALIGN %gradient.sum.init.next = add i32 %gradient.sum.init, 1 br label %gradient.sum.init.loop gradient.tile.loop:
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
%gradient.a.row = udiv i32 %gradient.a.global, %out.length %gradient.a.position = urem i32 %gradient.a.global, %out.length %gradient.a.row.base = mul i32 %gradient.a.row, %in.elements %gradient.a.term = add i32 %gradient.m.base, %gradient.a.m
%gradient.a.tile.index = call i32 @contraction_a_index(i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br i1 %gradient.a.vector, label %gradient.load.a.vector, label %gradient.load.a.scalar
gradient.load.a.vector:
%gradient.a.vector.index = add i32 %gradient.a.row.base, %gradient.a.term
%gradient.a.vector.source = getelementptr inbounds double, ptr addrspace(1) %input, i32 %gradient.a.vector.index
%gradient.a.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %gradient.a.vector.source, align 8
call void @contraction_stage_a_columns(<RECIPE_FRAGMENT_K x double> %gradient.a.vector.value, i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.a.scalar:
%gradient.a.value = call double @contraction_input( ptr addrspace(1) %input, i32 %gradient.a.row.base, i32 %gradient.a.position, i32 %gradient.a.term, i32 %span, i32 %in.length, i1 %is.conv )
br label %gradient.load.store
gradient.load.b.step: %gradient.b.local = sub i32 %gradient.load, %gradient.a.count %gradient.b.r = udiv i32 %gradient.b.local, %gradient.b.columns %gradient.b.column = urem i32 %gradient.b.local, %gradient.b.columns %gradient.b.n = mul i32 %gradient.b.column, %gradient.b.width %gradient.b.global = add i32 %gradient.r.base, %gradient.b.r
%gradient.b.row = udiv i32 %gradient.b.global, %out.length %gradient.b.position = urem i32 %gradient.b.global, %out.length %gradient.b.filter = add i32 %gradient.n.base, %gradient.b.n
%gradient.b.row.base = mul i32 %gradient.b.row, %out.elements %gradient.b.filter.base = mul i32 %gradient.b.filter, %out.length %gradient.b.local.index = add i32 %gradient.b.filter.base, %gradient.b.position %gradient.b.index = add i32 %gradient.b.row.base, %gradient.b.local.index
%gradient.b.tile.base = mul i32 %gradient.tile.m, %gradient.tile.k
%gradient.b.tile.local = call i32 @contraction_b_index(i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.n, i32 %gradient.tile.k) %gradient.b.tile.index = add i32 %gradient.b.tile.base, %gradient.b.tile.local
br i1 %gradient.b.vector, label %gradient.load.b.vector, label %gradient.load.b.scalar
gradient.load.b.vector:
%gradient.b.vector.value = call <RECIPE_FRAGMENT_K x double> @contraction_delta_vector16(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %gradient.b.index, i1 %relu)
call void @contraction_stage_b_fragment(<RECIPE_FRAGMENT_K x double> %gradient.b.vector.value, i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.b.scalar:
%gradient.b.value = call double @contraction_delta(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %gradient.b.index, i1 %relu)
br label %gradient.load.store
gradient.load.store: %gradient.load.value = phi double [ %gradient.a.value, %gradient.load.a.scalar ], [ %gradient.b.value, %gradient.load.b.scalar ] %gradient.load.index = phi i32 [ %gradient.a.tile.index, %gradient.load.a.scalar ], [ %gradient.b.tile.index, %gradient.load.b.scalar ]
%gradient.load.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %gradient.load.index store double %gradient.load.value, ptr addrspace(3) %gradient.load.ptr, align 8
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
call void @recipe.local.barrier()
br label %gradient.scalar.ready
gradient.scalar.ready:
call void @contraction_bias_accumulate(ptr addrspace(5) %bias.sums, ptr addrspace(1) %gradient.destination, i1 %gradient.bias.enable, i1 %gradient.r.first.tile, i1 %gradient.r.last.tile, i32 %lid, i32 %block, i32 %gradient.n.base, i32 %gradient.n.count, i32 %gradient.r.count, i32 %out.channels, i32 %window, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %gradient.store.offset)
call void @contraction_product_accumulate(ptr addrspace(5) %sums, i1 %gradient.lane.active, i1 %gradient.method.store, i32 %lid, i32 %gradient.lane.k, i32 %gradient.k.lanes, i32 %gradient.output.lane, i32 %gradient.output.lanes, i32 %gradient.output.m.base, i32 %gradient.output.n.base, i32 %gradient.m.count, i32 %gradient.n.count, i32 %gradient.r.count, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.product.done
gradient.product.done:
call void @recipe.local.barrier()
br i1 %gradient.r.more, label %gradient.tile.done, label %gradient.store.loop gradient.tile.done: br label %gradient.tile.loop
gradient.store.loop:
%gradient.store.register = phi i32 [ 0, %gradient.product.done ], [ %gradient.store.register.next, %gradient.store.next ] %gradient.store.more = icmp ult i32 %gradient.store.register, RECIPE_REGISTER_COUNT br i1 %gradient.store.more, label %gradient.store.test, label %gradient.bias.store.test
gradient.store.test: %gradient.store.register.m = urem i32 %gradient.store.register, RECIPE_REGISTER_M %gradient.store.register.n = udiv i32 %gradient.store.register, RECIPE_REGISTER_M %gradient.store.output.m.raw = call i32 @contraction_output_m(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.output.n.raw = call i32 @contraction_output_n(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.register.valid = call i1 @contraction_output_register_valid(i32 %gradient.store.register)
%gradient.store.output.m.valid = icmp ult i32 %gradient.store.output.m.raw, %gradient.m.count %gradient.store.output.n.valid = icmp ult i32 %gradient.store.output.n.raw, %gradient.n.count %gradient.store.output.valid = and i1 %gradient.store.output.m.valid, %gradient.store.output.n.valid %gradient.store.lane.active = and i1 %gradient.method.store, %gradient.store.output.valid %gradient.store.active = and i1 %gradient.store.lane.active, %gradient.store.register.valid br i1 %gradient.store.active, label %gradient.store, label %gradient.store.next
gradient.store: %gradient.store.filter = add i32 %gradient.n.base, %gradient.store.output.n.raw %gradient.store.term = add i32 %gradient.m.base, %gradient.store.output.m.raw %gradient.store.filter.base = mul i32 %gradient.store.filter, %window %gradient.store.local = add i32 %gradient.store.filter.base, %gradient.store.term %gradient.store.index = add i32 %gradient.store.offset, %gradient.store.local
%gradient.store.ptr = getelementptr inbounds double, ptr addrspace(1) %gradient.destination, i32 %gradient.store.index %gradient.store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %gradient.store.register %gradient.store.sum.wide = load RECIPE_STATE, ptr addrspace(5) %gradient.store.sum.ptr, align RECIPE_STATE_ALIGN %gradient.store.sum = call double @recipe.encode(RECIPE_STATE %gradient.store.sum.wide) store double %gradient.store.sum, ptr addrspace(1) %gradient.store.ptr, align 8
br label %gradient.store.next
gradient.store.next: %gradient.store.register.next = add i32 %gradient.store.register, 1 br label %gradient.store.loop gradient.bias.store.test:
br label %gradient.job.done
gradient.job.done: %gradient.task.next = add i32 %gradient.task, %groups br label %gradient.job.loop
gradient.finish:
br i1 %gradient.direct, label %previous.test, label %gradient.reduce.entry
gradient.reduce.entry:
call void @grid_barrier(i32 %threads)
call void @reduce_rows(ptr addrspace(1) %gradient.scratch, ptr addrspace(1) %gradient, i32 %gradient.splits, i32 %gradient.values, i32 %gradient.stride, i32 0, i32 %offset, i32 %threads)
br label %previous.test
previous.test: br i1 %write.input, label %previous.entry, label %exit previous.entry:
%previous.m.total = mul i32 %rows, %in.length %previous.r.total = mul i32 %out.channels, %span
%previous.m.short = icmp ult i32 %previous.tile.m, %previous.m.total %previous.m.tile = select i1 %previous.m.short, i32 %previous.tile.m, i32 %previous.m.total %previous.n.short = icmp ult i32 %previous.tile.n, %in.channels %previous.n.tile = select i1 %previous.n.short, i32 %previous.tile.n, i32 %in.channels
%previous.k.short = icmp ult i32 %previous.tile.k, %previous.r.total %previous.k.tile = select i1 %previous.k.short, i32 %previous.tile.k, i32 %previous.r.total
%previous.m.adjusted = add i32 %previous.m.total, %previous.m.tile %previous.m.numerator = sub i32 %previous.m.adjusted, 1 %previous.m.tiles = udiv i32 %previous.m.numerator, %previous.m.tile %previous.n.adjusted = add i32 %in.channels, %previous.n.tile %previous.n.numerator = sub i32 %previous.n.adjusted, 1 %previous.n.tiles = udiv i32 %previous.n.numerator, %previous.n.tile
%previous.jobs = mul i32 %previous.m.tiles, %previous.n.tiles br label %previous.job.loop previous.job.loop:
%previous.job = phi i32 [ %group, %previous.entry ], [ %previous.job.next, %previous.job.done ] %previous.job.more = icmp ult i32 %previous.job, %previous.jobs br i1 %previous.job.more, label %previous.job.step, label %exit
previous.job.step: %previous.m.group.short = icmp ult i32 %previous.m.tiles, RECIPE_CONTRACTION_SWIZZLE_M %previous.m.group.limit = select i1 %previous.m.group.short, i32 %previous.m.tiles, i32 RECIPE_CONTRACTION_SWIZZLE_M %previous.group.width = mul i32 %previous.m.group.limit, %previous.n.tiles %previous.group.index = udiv i32 %previous.job, %previous.group.width %previous.m.group.base = mul i32 %previous.group.index, %previous.m.group.limit %previous.m.group.remaining = sub i32 %previous.m.tiles, %previous.m.group.base %previous.m.group.tail = icmp ult i32 %previous.m.group.remaining, %previous.m.group.limit %previous.m.group.count = select i1 %previous.m.group.tail, i32 %previous.m.group.remaining, i32 %previous.m.group.limit %previous.group.local = urem i32 %previous.job, %previous.group.width %previous.m.group.local = urem i32 %previous.group.local, %previous.m.group.count %previous.m.index = add i32 %previous.m.group.base, %previous.m.group.local %previous.n.index = udiv i32 %previous.group.local, %previous.m.group.count %previous.m.base = mul i32 %previous.m.index, %previous.m.tile %previous.n.base = mul i32 %previous.n.index, %previous.n.tile
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
%previous.method.store = call i1 @contraction_store_lane(i1 %previous.lane.store, i32 %lid)
%previous.lane.n = udiv i32 %previous.output.lane, %previous.m.lanes %previous.lane.m = urem i32 %previous.output.lane, %previous.m.lanes
%previous.output.m.base = mul i32 %previous.lane.m, RECIPE_REGISTER_M %previous.output.n.base = mul i32 %previous.lane.n, RECIPE_REGISTER_N br label %previous.sum.init.loop previous.sum.init.loop:
%previous.sum.init = phi i32 [ 0, %previous.job.step ], [ %previous.sum.init.next, %previous.sum.init.step ] %previous.sum.init.more = icmp ult i32 %previous.sum.init, RECIPE_REGISTER_COUNT br i1 %previous.sum.init.more, label %previous.sum.init.step, label %previous.tile.loop
previous.sum.init.step: %previous.sum.init.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %previous.sum.init store RECIPE_STATE %state.zero, ptr addrspace(5) %previous.sum.init.ptr, align RECIPE_STATE_ALIGN %previous.sum.init.next = add i32 %previous.sum.init, 1 br label %previous.sum.init.loop previous.tile.loop:
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
previous.load.a.step: %previous.a.m = udiv i32 %previous.load, %previous.a.columns %previous.a.column = urem i32 %previous.load, %previous.a.columns %previous.a.r = mul i32 %previous.a.column, %previous.a.width %previous.a.term = add i32 %previous.r.base, %previous.a.r
%previous.a.filter = udiv i32 %previous.a.term, %span %previous.a.kernel = urem i32 %previous.a.term, %span %previous.a.global = add i32 %previous.m.base, %previous.a.m %previous.a.row = udiv i32 %previous.a.global, %in.length %previous.a.position = urem i32 %previous.a.global, %in.length
%previous.a.low = icmp uge i32 %previous.a.position, %previous.a.kernel %previous.a.position.raw = sub i32 %previous.a.position, %previous.a.kernel %previous.a.high = icmp ult i32 %previous.a.position.raw, %out.length %previous.a.valid = and i1 %previous.a.low, %previous.a.high
%previous.a.position.safe = select i1 %previous.a.valid, i32 %previous.a.position.raw, i32 0 %previous.a.row.base = mul i32 %previous.a.row, %out.elements %previous.a.filter.base = mul i32 %previous.a.filter, %out.length
%previous.a.local = add i32 %previous.a.filter.base, %previous.a.position.safe %previous.a.index = add i32 %previous.a.row.base, %previous.a.local %previous.a.tile.index = call i32 @contraction_a_index(i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br i1 %previous.a.vector, label %previous.load.a.vector, label %previous.load.a.scalar
previous.load.a.vector:
%previous.a.vector.delta = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %previous.a.index
%previous.a.vector.output = getelementptr inbounds double, ptr addrspace(1) %output, i32 %previous.a.index
%previous.a.vector.delta.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %previous.a.vector.delta, align 8
%previous.a.vector.output.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %previous.a.vector.output, align 8
call void @contraction_stage_delta_a_fragment(<RECIPE_FRAGMENT_K x double> %previous.a.vector.delta.value, <RECIPE_FRAGMENT_K x double> %previous.a.vector.output.value, i1 %relu, i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.a.scalar:
%previous.a.raw = call double @contraction_delta(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %previous.a.index, i1 %relu)
%previous.a.value = select i1 %previous.a.valid, double %previous.a.raw, double 0.0
br label %previous.load.store
previous.load.b.step: %previous.b.local = sub i32 %previous.load, %previous.a.count %previous.b.r = udiv i32 %previous.b.local, %previous.b.columns %previous.b.column = urem i32 %previous.b.local, %previous.b.columns %previous.b.n = mul i32 %previous.b.column, %previous.b.width %previous.b.term = add i32 %previous.r.base, %previous.b.r
%previous.b.filter = udiv i32 %previous.b.term, %span %previous.b.kernel = urem i32 %previous.b.term, %span %previous.b.channel = add i32 %previous.n.base, %previous.b.n %previous.b.filter.base = mul i32 %previous.b.filter, %window
%previous.b.channel.base = mul i32 %previous.b.channel, %span %previous.b.channel.local = add i32 %previous.b.channel.base, %previous.b.kernel %previous.b.index = add i32 %previous.b.filter.base, %previous.b.channel.local
%previous.b.tile.base = mul i32 %previous.tile.m, %previous.tile.k
%previous.b.tile.local = call i32 @contraction_b_index(i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.n, i32 %previous.tile.k) %previous.b.tile.index = add i32 %previous.b.tile.base, %previous.b.tile.local
br i1 %previous.b.vector, label %previous.load.b.vector, label %previous.load.b.scalar
previous.load.b.vector:
%previous.b.vector.source = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %previous.b.index
%previous.b.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %previous.b.vector.source, align 8
call void @contraction_stage_b_fragment(<RECIPE_FRAGMENT_K x double> %previous.b.vector.value, i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.b.scalar:
%previous.b.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %previous.b.index
%previous.b.value = load double, ptr addrspace(1) %previous.b.ptr, align 8
br label %previous.load.store
previous.load.store: %previous.load.value = phi double [ %previous.a.value, %previous.load.a.scalar ], [ %previous.b.value, %previous.load.b.scalar ] %previous.load.index = phi i32 [ %previous.a.tile.index, %previous.load.a.scalar ], [ %previous.b.tile.index, %previous.load.b.scalar ]
%previous.load.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %previous.load.index store double %previous.load.value, ptr addrspace(3) %previous.load.ptr, align 8
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
call void @recipe.local.barrier()
call void @contraction_product_accumulate(ptr addrspace(5) %sums, i1 %previous.lane.active, i1 %previous.method.store, i32 %lid, i32 %previous.lane.k, i32 %previous.k.lanes, i32 %previous.output.lane, i32 %previous.lanes, i32 %previous.output.m.base, i32 %previous.output.n.base, i32 %previous.m.count, i32 %previous.n.count, i32 %previous.r.count, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k) call void @recipe.local.barrier()
%previous.r.next = add i32 %previous.r.base, %previous.r.count %previous.r.more = icmp ult i32 %previous.r.next, %previous.r.total br i1 %previous.r.more, label %previous.tile.done, label %previous.store.loop previous.tile.done: br label %previous.tile.loop previous.store.loop:
%previous.store.register = phi i32 [ 0, %previous.load.ready ], [ %previous.store.register.next, %previous.store.next ] %previous.store.more = icmp ult i32 %previous.store.register, RECIPE_REGISTER_COUNT br i1 %previous.store.more, label %previous.store.test, label %previous.job.done
previous.store.test: %previous.store.output.m.raw = call i32 @contraction_output_m(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.output.n.raw = call i32 @contraction_output_n(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.register.valid = call i1 @contraction_output_register_valid(i32 %previous.store.register)
%previous.store.output.m.valid = icmp ult i32 %previous.store.output.m.raw, %previous.m.count %previous.store.output.n.valid = icmp ult i32 %previous.store.output.n.raw, %previous.n.count %previous.store.output.valid = and i1 %previous.store.output.m.valid, %previous.store.output.n.valid %previous.lane.output.active = and i1 %previous.method.store, %previous.store.output.valid %previous.store.active = and i1 %previous.lane.output.active, %previous.store.register.valid br i1 %previous.store.active, label %previous.store, label %previous.store.next
previous.store: %previous.store.m.global = add i32 %previous.m.base, %previous.store.output.m.raw %previous.store.channel = add i32 %previous.n.base, %previous.store.output.n.raw %previous.store.row = udiv i32 %previous.store.m.global, %in.length %previous.store.position = urem i32 %previous.store.m.global, %in.length
%previous.store.row.base = mul i32 %previous.store.row, %in.elements %previous.store.channel.base = mul i32 %previous.store.channel, %in.length %previous.store.local = add i32 %previous.store.channel.base, %previous.store.position %previous.store.index = add i32 %previous.store.row.base, %previous.store.local %previous.store.ptr = getelementptr inbounds double, ptr addrspace(1) %previous, i32 %previous.store.index
%previous.store.old = load double, ptr addrspace(1) %previous.store.ptr, align 8 %previous.store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %previous.store.register %previous.store.sum.wide = load RECIPE_STATE, ptr addrspace(5) %previous.store.sum.ptr, align RECIPE_STATE_ALIGN %previous.store.sum = call double @recipe.encode(RECIPE_STATE %previous.store.sum.wide) %previous.store.value = call double @recipe.add(double %previous.store.old, double %previous.store.sum) store double %previous.store.value, ptr addrspace(1) %previous.store.ptr, align 8 br label %previous.store.next
previous.store.next: %previous.store.register.next = add i32 %previous.store.register, 1 br label %previous.store.loop previous.job.done: %previous.job.next = add i32 %previous.job, %groups br label %previous.job.loop exit: ret void }
define internal void @scan_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output,
ptr addrspace(1) %context, ptr addrspace(1) %delta, ptr addrspace(1) %previous,
ptr addrspace(1) %gradient, i1 %write.input, i32 %rows, i32 %in.channels,
i32 %length, i32 %out.channels, i32 %gates, i32 %parameters, i32 %offset,
i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) #3 { entry:
%tid = call i32 @llvm.amdgcn.workitem.id.x() %in.elements = mul i32 %in.channels, %length
%out.elements = mul i32 %out.channels, %length %batch = mul i32 %rows, %out.elements
%gate.stride.0 = mul i32 %in.channels, %out.channels %state.matrix = mul i32 %out.channels, %out.channels
%gate.stride.1 = add i32 %gate.stride.0, %state.matrix %gate.stride = add i32 %gate.stride.1, %out.channels
%delta.base.factor = add i32 %gates, 1 %delta.base = mul i32 %delta.base.factor, %batch %gate2.batch = mul i32 %batch, 2
%row.gradient.factor = mul i32 %gates, 2 %row.gradient.factor.1 = add i32 %row.gradient.factor, 1
%row.gradient.base = mul i32 %row.gradient.factor.1, %batch %rnn = icmp eq i32 %gates, 1
%gru = icmp eq i32 %gates, 3 %lstm = icmp eq i32 %gates, 4 %simple = or i1 %rnn, %gru
%supported = or i1 %simple, %lstm br i1 %supported, label %row.loop, label %invalid row.loop:
%row = phi i32 [ %tid, %entry ], [ %row.next, %row.done ]
%row.more = icmp ult i32 %row, %rows br i1 %row.more, label %clear.gradient.loop, label %reduce.entry
clear.gradient.loop: %clear.p = phi i32 [ 0, %row.loop ], [ %clear.next, %clear.gradient.step ]
%row.gradient.offset = mul i32 %row, %parameters %row.gradient.start = add i32 %row.gradient.base, %row.gradient.offset
%clear.more = icmp ult i32 %clear.p, %parameters br i1 %clear.more, label %clear.gradient.step, label %clear.state.loop
clear.gradient.step: %clear.index = add i32 %row.gradient.start, %clear.p
%clear.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %clear.index
store double 0.0, ptr addrspace(1) %clear.ptr, align 8 %clear.next = add nuw i32 %clear.p, 1
br label %clear.gradient.loop clear.state.loop:
%clear.h = phi i32 [ 0, %clear.gradient.loop ], [ %clear.h.next, %clear.state.step ]
%scratch.base.0 = mul i32 %rows, %parameters %scratch.base = add i32 %row.gradient.base, %scratch.base.0
%scratch.row = mul i32 %row, %out.channels %dh.start = add i32 %scratch.base, %scratch.row
%dc.base.0 = mul i32 %rows, %out.channels %dc.base = add i32 %scratch.base, %dc.base.0
%dc.start = add i32 %dc.base, %scratch.row %clear.h.more = icmp ult i32 %clear.h, %out.channels
br i1 %clear.h.more, label %clear.state.step, label %time.loop clear.state.step:
%clear.dh.index = add i32 %dh.start, %clear.h %clear.dc.index = add i32 %dc.start, %clear.h
%clear.dh.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %clear.dh.index
%clear.dc.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %clear.dc.index
store double 0.0, ptr addrspace(1) %clear.dh.ptr, align 8 store double 0.0, ptr addrspace(1) %clear.dc.ptr, align 8
%clear.h.next = add nuw i32 %clear.h, 1 br label %clear.state.loop time.loop:
%time = phi i32 [ %length, %clear.state.loop ], [ %time.current, %time.done ] %time.current = sub i32 %time, 1
%row.output.base = mul i32 %row, %out.elements %input.row.base = mul i32 %row, %in.elements
%previous.time = sub i32 %time.current, 1 %previous.exists = icmp sge i32 %previous.time, 0
%previous.safe = select i1 %previous.exists, i32 %previous.time, i32 0 %time.more = icmp sge i32 %time.current, 0
br i1 %time.more, label %scan.mode, label %row.done scan.mode:
br i1 %lstm, label %gate.delta.loop, label %rnn.test rnn.test:
br i1 %rnn, label %rnn.delta.loop, label %gru.delta.loop rnn.delta.loop:
%rnn.hidden = phi i32 [ 0, %rnn.test ], [ %rnn.next, %rnn.delta.step ]
%rnn.more = icmp ult i32 %rnn.hidden, %out.channels
br i1 %rnn.more, label %rnn.delta.step, label %delta.done rnn.delta.step:
%rnn.hidden.base = mul i32 %rnn.hidden, %length %rnn.local = add i32 %rnn.hidden.base, %time.current
%rnn.index = add i32 %row.output.base, %rnn.local %rnn.dy.ptr = getelementptr inbounds double,
ptr addrspace(1) %delta, i32 %rnn.index %rnn.future.index = add i32 %dh.start, %rnn.hidden
%rnn.future.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %rnn.future.index
%rnn.gate.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %rnn.index
%rnn.dy = load double, ptr addrspace(1) %rnn.dy.ptr, align 8
%rnn.future = load double, ptr addrspace(1) %rnn.future.ptr, align 8
%rnn.gate = load double, ptr addrspace(1) %rnn.gate.ptr, align 8 %rnn.dh = call double @recipe.add(double %rnn.dy, double %rnn.future)
%rnn.square = call double @recipe.mul(double %rnn.gate, double %rnn.gate) %rnn.derivative = call double @recipe.sub(double 1.0, double %rnn.square)
%rnn.delta = call double @recipe.mul(double %rnn.dh, double %rnn.derivative) %rnn.delta.index = add i32 %delta.base, %rnn.index
%rnn.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %rnn.delta.index
store double %rnn.delta, ptr addrspace(1) %rnn.delta.ptr, align 8 %rnn.next = add i32 %rnn.hidden, 1
br label %rnn.delta.loop gru.delta.loop: %gru.hidden = phi i32 [ 0, %rnn.test ], [ %gru.next, %gru.delta.step ]
%gru.more = icmp ult i32 %gru.hidden, %out.channels
br i1 %gru.more, label %gru.delta.step, label %gru.reset.loop gru.delta.step:
%gru.hidden.base = mul i32 %gru.hidden, %length %gru.local = add i32 %gru.hidden.base, %time.current
%gru.index = add i32 %row.output.base, %gru.local %gru.previous.local = add i32 %gru.hidden.base, %previous.safe
%gru.previous.index = add i32 %row.output.base, %gru.previous.local
%gru.dy.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %gru.index
%gru.future.index = add i32 %dh.start, %gru.hidden
%gru.future.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.future.index
%gru.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %gru.previous.index
%gru.z.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.index
%gru.n.index = add i32 %gru.index, %gate2.batch
%gru.n.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.n.index
%gru.dy = load double, ptr addrspace(1) %gru.dy.ptr, align 8
%gru.future = load double, ptr addrspace(1) %gru.future.ptr, align 8
%gru.previous.loaded = load double, ptr addrspace(1) %gru.previous.ptr, align 8
%gru.previous = select i1 %previous.exists, double %gru.previous.loaded, double 0.0
%gru.z = load double, ptr addrspace(1) %gru.z.ptr, align 8
%gru.n = load double, ptr addrspace(1) %gru.n.ptr, align 8 %gru.dh = call double @recipe.add(double %gru.dy, double %gru.future)
%gru.one.z = call double @recipe.sub(double 1.0, double %gru.z) %gru.z.difference = call double @recipe.sub(double %gru.previous, double %gru.n)
%gru.dz.0 = call double @recipe.mul(double %gru.dh, double %gru.z.difference) %gru.dz.1 = call double @recipe.mul(double %gru.dz.0, double %gru.z)
%gru.dz = call double @recipe.mul(double %gru.dz.1, double %gru.one.z) %gru.n.square = call double @recipe.mul(double %gru.n, double %gru.n)
%gru.n.derivative = call double @recipe.sub(double 1.0, double %gru.n.square) %gru.dn.0 = call double @recipe.mul(double %gru.dh, double %gru.one.z)
%gru.dn = call double @recipe.mul(double %gru.dn.0, double %gru.n.derivative) %gru.dz.index = add i32 %delta.base, %gru.index
%gru.dn.index.0 = add i32 %delta.base, %gate2.batch %gru.dn.index = add i32 %gru.dn.index.0, %gru.index
%gru.dz.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.dz.index
%gru.dn.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.dn.index
store double %gru.dz, ptr addrspace(1) %gru.dz.ptr, align 8
store double %gru.dn, ptr addrspace(1) %gru.dn.ptr, align 8 %gru.next = add i32 %gru.hidden, 1
br label %gru.delta.loop gru.reset.loop:
%gru.source = phi i32 [ 0, %gru.delta.loop ], [ %gru.source.next, %gru.reset.store ]
%gru.source.more = icmp ult i32 %gru.source, %out.channels
br i1 %gru.source.more, label %gru.reset.sum.loop, label %delta.done gru.reset.sum.loop:
%gru.target = phi i32 [ 0, %gru.reset.loop ], [ %gru.target.next, %gru.reset.sum.step ]
%gru.reset.sum = phi double [ 0.0, %gru.reset.loop ], [ %gru.reset.sum.next, %gru.reset.sum.step ]
%gru.target.more = icmp ult i32 %gru.target, %out.channels
br i1 %gru.target.more, label %gru.reset.sum.step, label %gru.reset.store gru.reset.sum.step:
%gru.candidate.base = mul i32 %gate.stride, 2 %gru.candidate.state = add i32 %gru.candidate.base, %gate.stride.0
%gru.weight.row = mul i32 %gru.source, %out.channels %gru.weight.local = add i32 %gru.weight.row, %gru.target
%gru.weight.index = add i32 %gru.candidate.state, %gru.weight.local
%gru.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %gru.weight.index
%gru.target.base = mul i32 %gru.target, %length %gru.target.local = add i32 %gru.target.base, %time.current
%gru.target.index = add i32 %row.output.base, %gru.target.local %gru.target.delta.0 = add i32 %delta.base, %gate2.batch
%gru.target.delta.index = add i32 %gru.target.delta.0, %gru.target.index
%gru.target.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.target.delta.index
%gru.weight = load double, ptr addrspace(1) %gru.weight.ptr, align 8
%gru.target.delta = load double, ptr addrspace(1) %gru.target.delta.ptr, align 8
%gru.reset.product = call double @recipe.mul(double %gru.weight, double %gru.target.delta)
%gru.reset.sum.next = call double @recipe.add(double %gru.reset.sum, double %gru.reset.product)
%gru.target.next = add i32 %gru.target, 1 br label %gru.reset.sum.loop gru.reset.store:
%gru.source.base = mul i32 %gru.source, %length %gru.source.local = add i32 %gru.source.base, %time.current
%gru.source.index = add i32 %row.output.base, %gru.source.local
%gru.source.previous.local = add i32 %gru.source.base, %previous.safe
%gru.source.previous.index = add i32 %row.output.base, %gru.source.previous.local
%gru.source.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %gru.source.previous.index
%gru.r.index = add i32 %batch, %gru.source.index
%gru.r.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.r.index
%gru.source.previous.loaded = load double, ptr addrspace(1) %gru.source.previous.ptr, align 8
%gru.source.previous = select i1 %previous.exists, double %gru.source.previous.loaded, double 0.0
%gru.r = load double, ptr addrspace(1) %gru.r.ptr, align 8
%gru.dr = call double @recipe.mul(double %gru.reset.sum, double %gru.source.previous) %gru.one.r = call double @recipe.sub(double 1.0, double %gru.r)
%gru.dr.0 = call double @recipe.mul(double %gru.dr, double %gru.r) %gru.dr.1 = call double @recipe.mul(double %gru.dr.0, double %gru.one.r)
%gru.dr.base = add i32 %delta.base, %batch %gru.dr.index = add i32 %gru.dr.base, %gru.source.index
%gru.dr.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %gru.dr.index
store double %gru.dr.1, ptr addrspace(1) %gru.dr.ptr, align 8 %gru.source.next = add i32 %gru.source, 1
br label %gru.reset.loop gate.delta.loop: %hidden = phi i32 [ 0, %scan.mode ], [ %hidden.next, %gate.delta.step ]
%hidden.more = icmp ult i32 %hidden, %out.channels br i1 %hidden.more, label %gate.delta.step, label %delta.done
gate.delta.step: %hidden.base = mul i32 %hidden, %length %local = add i32 %hidden.base, %time.current
%index = add i32 %row.output.base, %local %previous.local = add i32 %hidden.base, %previous.safe
%previous.index = add i32 %row.output.base, %previous.local %cell.base = mul i32 %gates, %batch
%cell.index = add i32 %cell.base, %index %cell.previous.index = add i32 %cell.base, %previous.index
%dy.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %index %dh.index = add i32 %dh.start, %hidden
%dc.index = add i32 %dc.start, %hidden %dh.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %dh.index
%dc.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %dc.index
%cell.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %cell.index
%cell.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %cell.previous.index
%dy = load double, ptr addrspace(1) %dy.ptr, align 8 %dh.future = load double, ptr addrspace(1) %dh.ptr, align 8
%dc.future = load double, ptr addrspace(1) %dc.ptr, align 8 %cell = load double, ptr addrspace(1) %cell.ptr, align 8
%cell.previous.loaded = load double, ptr addrspace(1) %cell.previous.ptr, align 8
%cell.previous = select i1 %previous.exists, double %cell.previous.loaded, double 0.0
%i.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %index %f.index = add i32 %batch, %index
%o.index = add i32 %f.index, %batch %g.index = add i32 %o.index, %batch
%f.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %f.index
%o.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %o.index
%g.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %g.index
%i = load double, ptr addrspace(1) %i.ptr, align 8 %f = load double, ptr addrspace(1) %f.ptr, align 8
%o = load double, ptr addrspace(1) %o.ptr, align 8 %g = load double, ptr addrspace(1) %g.ptr, align 8
%dh = call double @recipe.add(double %dy, double %dh.future) %cell.tanh = call double @recipe.tanh(double %cell)
%cell.tanh.square = call double @recipe.mul(double %cell.tanh, double %cell.tanh) %cell.tanh.derivative = call double @recipe.sub(double 1.0, double %cell.tanh.square)
%cell.chain.0 = call double @recipe.mul(double %dh, double %o) %cell.chain = call double @recipe.mul(double %cell.chain.0, double %cell.tanh.derivative)
%dc = call double @recipe.add(double %dc.future, double %cell.chain) %one.o = call double @recipe.sub(double 1.0, double %o) %do.0 = call double @recipe.mul(double %dh, double %cell.tanh)
%do.1 = call double @recipe.mul(double %do.0, double %o) %do = call double @recipe.mul(double %do.1, double %one.o) %one.i = call double @recipe.sub(double 1.0, double %i) %di.0 = call double @recipe.mul(double %dc, double %g)
%di.1 = call double @recipe.mul(double %di.0, double %i) %di = call double @recipe.mul(double %di.1, double %one.i) %one.f = call double @recipe.sub(double 1.0, double %f)
%df.0 = call double @recipe.mul(double %dc, double %cell.previous) %df.1 = call double @recipe.mul(double %df.0, double %f) %df = call double @recipe.mul(double %df.1, double %one.f)
%g.square = call double @recipe.mul(double %g, double %g) %one.g.square = call double @recipe.sub(double 1.0, double %g.square) %dg.0 = call double @recipe.mul(double %dc, double %i)
%dg = call double @recipe.mul(double %dg.0, double %one.g.square) %dc.previous = call double @recipe.mul(double %dc, double %f)
store double %dc.previous, ptr addrspace(1) %dc.ptr, align 8 %delta0.index = add i32 %delta.base, %index
%delta1.index = add i32 %delta0.index, %batch %delta2.index = add i32 %delta1.index, %batch
%delta3.index = add i32 %delta2.index, %batch
%delta0.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %delta0.index
%delta1.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %delta1.index
%delta2.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %delta2.index
%delta3.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %delta3.index
store double %di, ptr addrspace(1) %delta0.ptr, align 8 store double %df, ptr addrspace(1) %delta1.ptr, align 8
store double %do, ptr addrspace(1) %delta2.ptr, align 8 store double %dg, ptr addrspace(1) %delta3.ptr, align 8
%hidden.next = add nuw i32 %hidden, 1 br label %gate.delta.loop delta.done: br label %parameter.loop parameter.loop:
%p = phi i32 [ 0, %delta.done ], [ %p.next, %parameter.advance ] %p.more = icmp ult i32 %p, %parameters
br i1 %p.more, label %parameter.step, label %hidden.gradient.loop parameter.step:
%gate = udiv i32 %p, %gate.stride %within = urem i32 %p, %gate.stride
%input.weight = icmp ult i32 %within, %gate.stride.0
br i1 %input.weight, label %parameter.advance, label %parameter.value parameter.value:
%state.end = add i32 %gate.stride.0, %state.matrix %state.weight = icmp ult i32 %within, %state.end
%state.index = sub i32 %within, %gate.stride.0 %selected.index = select i1 %state.weight, i32 %state.index, i32 0
%source.channel = udiv i32 %selected.index, %out.channels %target.hidden = urem i32 %selected.index, %out.channels
%bias.hidden = sub i32 %within, %state.end %delta.hidden = select i1 %state.weight, i32 %target.hidden, i32 %bias.hidden
%delta.hidden.base = mul i32 %delta.hidden, %length %delta.local = add i32 %delta.hidden.base, %time.current
%delta.row.local = add i32 %row.output.base, %delta.local %delta.gate.base = mul i32 %gate, %batch
%delta.gate.local = add i32 %delta.base, %delta.gate.base %delta.index = add i32 %delta.gate.local, %delta.row.local
%gate.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %delta.index
%gate.delta = load double, ptr addrspace(1) %gate.delta.ptr, align 8
%state.hidden.base = mul i32 %source.channel, %length
%state.local = add i32 %state.hidden.base, %previous.safe %state.index.value = add i32 %row.output.base, %state.local
%state.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i32 %state.index.value
%state.loaded = load double, ptr addrspace(1) %state.ptr, align 8
%state.value = select i1 %previous.exists, double %state.loaded, double 0.0
%candidate.gate = icmp eq i32 %gate, 2 %gru.candidate = and i1 %gru, %candidate.gate
%parameter.reset.local = add i32 %state.hidden.base, %time.current
%parameter.reset.row = add i32 %row.output.base, %parameter.reset.local
%parameter.reset.raw = add i32 %batch, %parameter.reset.row
%parameter.reset.index = select i1 %gru.candidate, i32 %parameter.reset.raw, i32 0
%parameter.reset.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %parameter.reset.index
%parameter.reset = load double, ptr addrspace(1) %parameter.reset.ptr, align 8
%parameter.reset.state = call double @recipe.mul(double %parameter.reset, double %state.value)
%parameter.state = select i1 %gru.candidate, double %parameter.reset.state, double %state.value
%source.value = select i1 %state.weight, double %parameter.state, double 1.0
%contribution = call double @recipe.mul(double %source.value, double %gate.delta) %row.gradient.index = add i32 %row.gradient.start, %p
%row.gradient.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %row.gradient.index
%row.gradient.old = load double, ptr addrspace(1) %row.gradient.ptr, align 8
%row.gradient.new = call double @recipe.add(double %row.gradient.old, double %contribution)
store double %row.gradient.new, ptr addrspace(1) %row.gradient.ptr, align 8
br label %parameter.advance parameter.advance:
%p.next = add nuw i32 %p, 1 br label %parameter.loop hidden.gradient.loop:
%state.channel = phi i32 [ 0, %parameter.loop ], [ %state.channel.next, %hidden.gradient.store ]
%state.channel.more = icmp ult i32 %state.channel, %out.channels
br i1 %state.channel.more, label %hidden.gradient.sum.loop, label %time.done hidden.gradient.sum.loop:
%state.term = phi i32 [ 0, %hidden.gradient.loop ], [ %state.term.next, %hidden.gradient.sum.step ]
%state.sum = phi double [ 0.0, %hidden.gradient.loop ], [ %state.sum.next, %hidden.gradient.sum.step ]
%state.terms = mul i32 %gates, %out.channels %state.term.more = icmp ult i32 %state.term, %state.terms
br i1 %state.term.more, label %hidden.gradient.sum.step, label %hidden.gradient.store hidden.gradient.sum.step:
%state.gate = udiv i32 %state.term, %out.channels %state.hidden = urem i32 %state.term, %out.channels
%state.gate.base = mul i32 %state.gate, %gate.stride %state.matrix.base = add i32 %state.gate.base, %gate.stride.0
%state.weight.row = mul i32 %state.channel, %out.channels %state.weight.local = add i32 %state.weight.row, %state.hidden
%state.weight.index = add i32 %state.matrix.base, %state.weight.local
%state.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %state.weight.index
%state.delta.hidden.base = mul i32 %state.hidden, %length
%state.delta.local = add i32 %state.delta.hidden.base, %time.current
%state.delta.row = add i32 %row.output.base, %state.delta.local %state.delta.gate.base = mul i32 %state.gate, %batch
%state.delta.base = add i32 %delta.base, %state.delta.gate.base
%state.delta.index = add i32 %state.delta.base, %state.delta.row
%state.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %state.delta.index
%state.weight.value = load double, ptr addrspace(1) %state.weight.ptr, align 8
%state.delta.value = load double, ptr addrspace(1) %state.delta.ptr, align 8
%state.product = call double @recipe.mul(double %state.weight.value, double %state.delta.value) %state.candidate = icmp eq i32 %state.gate, 2
%state.gru.candidate = and i1 %gru, %state.candidate %state.reset.hidden.base = mul i32 %state.channel, %length
%state.reset.local = add i32 %state.reset.hidden.base, %time.current
%state.reset.row = add i32 %row.output.base, %state.reset.local %state.reset.raw = add i32 %batch, %state.reset.row
%state.reset.index = select i1 %state.gru.candidate, i32 %state.reset.raw, i32 0
%state.reset.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %state.reset.index
%state.reset = load double, ptr addrspace(1) %state.reset.ptr, align 8
%state.reset.product = call double @recipe.mul(double %state.product, double %state.reset)
%state.contribution = select i1 %state.gru.candidate, double %state.reset.product, double %state.product
%state.sum.next = call double @recipe.add(double %state.sum, double %state.contribution) %state.term.next = add nuw i32 %state.term, 1
br label %hidden.gradient.sum.loop hidden.gradient.store: %state.dh.index = add i32 %dh.start, %state.channel
%state.dh.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %state.dh.index
%state.direct.hidden.base = mul i32 %state.channel, %length
%state.direct.local = add i32 %state.direct.hidden.base, %time.current
%state.direct.index = add i32 %row.output.base, %state.direct.local
%state.direct.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i32 %state.direct.index
%state.direct.z.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i32 %state.direct.index
%state.direct.dy = load double, ptr addrspace(1) %state.direct.delta.ptr, align 8
%state.direct.future = load double, ptr addrspace(1) %state.dh.ptr, align 8
%state.direct.z = load double, ptr addrspace(1) %state.direct.z.ptr, align 8
%state.direct.dh = call double @recipe.add(double %state.direct.dy, double %state.direct.future)
%state.direct.raw = call double @recipe.mul(double %state.direct.z, double %state.direct.dh)
%state.direct = select i1 %gru, double %state.direct.raw, double 0.0
%state.total = call double @recipe.add(double %state.sum, double %state.direct)
store double %state.total, ptr addrspace(1) %state.dh.ptr, align 8 %state.channel.next = add nuw i32 %state.channel, 1
br label %hidden.gradient.loop time.done: br label %time.loop row.done: %row.next = add i32 %row, %threads
br label %row.loop reduce.entry: call void @llvm.amdgcn.s.barrier()
call void @reduce_rows(ptr addrspace(1) %context, ptr addrspace(1) %gradient, i32 %rows, i32 %parameters, i32 %parameters, i32 %row.gradient.base, i32 %offset, i32 %threads)
br label %projection.entry
projection.entry: call void @llvm.amdgcn.s.barrier() br label %projection.loop projection.loop:
%projection.gate = phi i32 [ 0, %projection.entry ], [ %projection.next, %projection.step ]
%projection.more = icmp ult i32 %projection.gate, %gates
br i1 %projection.more, label %projection.step, label %exit projection.step:
%projection.weight.offset = mul i32 %projection.gate, %gate.stride
%projection.weights = getelementptr inbounds double, ptr addrspace(1) %weights, i32 %projection.weight.offset
%projection.delta.gate = mul i32 %projection.gate, %batch
%projection.delta.offset = add i32 %delta.base, %projection.delta.gate
%projection.delta = getelementptr inbounds double, ptr addrspace(1) %context, i32 %projection.delta.offset
%projection.gradient.offset = add i32 %offset, %projection.weight.offset
call void @contraction_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %projection.weights, ptr addrspace(1) %output,
ptr addrspace(1) %projection.delta, ptr addrspace(1) %previous, ptr addrspace(1) %gradient, i1 %write.input, i1 false, i1 false, i1 false,
i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels, i32 %length, i32 0,
i32 %projection.gradient.offset, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k,
i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) call void @llvm.amdgcn.s.barrier() %projection.next = add i32 %projection.gate, 1 br label %projection.loop
invalid: call void @llvm.trap() br label %exit exit: ret void } attributes #0 = { nounwind "amdgpu-flat-work-group-size"="RECIPE_WORKGROUP_SIZE,RECIPE_WORKGROUP_SIZE" } attributes #1 = { alwaysinline nounwind } attributes #3 = { noinline nounwind }
; Fully unroll the product loop so each insertelement uses a constant lane index
; and the accumulator vector remains in registers.
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
