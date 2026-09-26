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
; RECIPE_WAVE_HELPERS
; RECIPE_BLOCK_HELPERS
declare void @llvm.trap() @contraction_tile = external addrspace(3) global [0 x double], align 16
define internal double @contraction_input(
ptr addrspace(1) %input, i64 %row.base, i32 %position, i32 %term, i32 %span, i32 %length, i1 %conv ) #1 { entry:
%channel = udiv i32 %term, %span %window = urem i32 %term, %span
%offset = select i1 %conv, i32 %window, i32 0 %channel.wide = zext i32 %channel to i64 %length.wide = zext i32 %length to i64 %position.wide = zext i32 %position to i64 %offset.wide = zext i32 %offset to i64
%channel.base = mul i64 %channel.wide, %length.wide
%local.0 = add i64 %channel.base, %position.wide %local = add i64 %local.0, %offset.wide
%index = add i64 %row.base, %local %ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %index
%value = load double, ptr addrspace(1) %ptr, align 8 ret double %value }

; The backward passes read the delta in the arithmetic type: nothing on the adjoint
; side is ever rounded to the model type. The relu gate still reads the model output.
define internal RECIPE_STATE @contraction_delta_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %index
%delta.value = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %index
%output.value = load double, ptr addrspace(1) %output.ptr, align 8
%positive = call i1 @recipe.ogt(double %output.value, double 0.0)
%activated = select i1 %positive, RECIPE_STATE %delta.value, RECIPE_STATE %zero
br label %done
done:
%value = phi RECIPE_STATE [ %delta.value, %entry ], [ %activated, %activation ]
ret RECIPE_STATE %value
}
define internal <16 x RECIPE_STATE> @contraction_delta_vector16_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %index
%delta.value = load <16 x RECIPE_STATE>, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %index
%output.value = load <16 x double>, ptr addrspace(1) %output.ptr, align 8
br label %activation.loop
activation.loop:
%activation.index = phi i32 [ 0, %activation ], [ %activation.next, %activation.step ]
%activation.values = phi <16 x RECIPE_STATE> [ zeroinitializer, %activation ], [ %activation.values.next, %activation.step ]
%activation.more = icmp ult i32 %activation.index, 16
br i1 %activation.more, label %activation.step, label %done
activation.step:
%activation.output = extractelement <16 x double> %output.value, i32 %activation.index
%activation.delta = extractelement <16 x RECIPE_STATE> %delta.value, i32 %activation.index
%activation.positive = call i1 @recipe.ogt(double %activation.output, double 0.0)
%activation.value = select i1 %activation.positive, RECIPE_STATE %activation.delta, RECIPE_STATE %zero
%activation.values.next = insertelement <16 x RECIPE_STATE> %activation.values, RECIPE_STATE %activation.value, i32 %activation.index
%activation.next = add i32 %activation.index, 1
br label %activation.loop
done:
%value = phi <16 x RECIPE_STATE> [ %delta.value, %entry ], [ %activation.values, %activation.loop ]
ret <16 x RECIPE_STATE> %value
}
define internal void @reduce_rows_state(ptr addrspace(1) %source, ptr addrspace(1) %target, i32 %rows, i32 %columns, i32 %stride, i32 %source.offset, i32 %target.offset, i32 %threads) #1 {
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
%target.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %target, i32 %target.index
%source.first.index = add i32 %source.offset, %parameter
%source.first.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %source, i32 %source.first.index
%source.first = load RECIPE_STATE, ptr addrspace(1) %source.first.ptr, align RECIPE_STATE_ALIGN
br label %row.loop
row.loop:
%row = phi i32 [ 1, %seed.load ], [ %row.next, %row.step ]
%sum = phi RECIPE_STATE [ %source.first, %seed.load ], [ %sum.next, %row.step ]
%row.more = icmp ult i32 %row, %rows
br i1 %row.more, label %row.step, label %store
row.step:
%row.base = mul i32 %row, %stride
%source.local = add i32 %row.base, %parameter
%source.index = add i32 %source.offset, %source.local
%source.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %source, i32 %source.index
%source.value = load RECIPE_STATE, ptr addrspace(1) %source.ptr, align RECIPE_STATE_ALIGN
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %source.value)
%row.next = add i32 %row, 1
br label %row.loop
store:
store RECIPE_STATE %sum, ptr addrspace(1) %target.ptr, align RECIPE_STATE_ALIGN
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
define internal void @contraction_zero_edges_bs(i32 %m.count, i32 %n.count, i32 %k.count, i32 %lid, i32 %block, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
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
%a.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %a.any
store double 0.0, ptr addrspace(3) %a.ptr, align 8
br label %store
store.b:
%b.any = phi i32 [ %b.index, %b.step ], [ %b.k.index, %b.k.step ]
%b.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %b.any
store RECIPE_STATE %state.zero, ptr addrspace(3) %b.ptr, align RECIPE_STATE_ALIGN
br label %store
store:
%next = add i32 %p, %block
br label %loop
exit:
ret void
}
define internal void @contraction_zero_edges_as(i32 %m.count, i32 %n.count, i32 %k.count, i32 %lid, i32 %block, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
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
%a.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %a.any
store RECIPE_STATE %state.zero, ptr addrspace(3) %a.ptr, align RECIPE_STATE_ALIGN
br label %store
store.b:
%b.any = phi i32 [ %b.index, %b.step ], [ %b.k.index, %b.k.step ]
%b.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %b.any
store double 0.0, ptr addrspace(3) %b.ptr, align 8
br label %store
store:
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

; Backward tiles hold the adjoint operand in RECIPE_STATE. The gradient pass keeps
; A (the input) in the model type at its usual index and puts B (the delta) in a
; state region that starts after the model A region; the previous pass puts A (the
; delta) in a state region at the tile's start and B (the weights) in the model
; type after it. Each base is whole elements of its region's own type, and the
; backward passes always take the vector layout.
define internal i32 @contraction_state_after_model(i32 %elements) #1 {
entry:
%bytes = mul i32 %elements, RECIPE_MODEL_BYTES
%pad = sub i32 RECIPE_STATE_ALIGN, 1
%rounded = add i32 %bytes, %pad
%base = udiv i32 %rounded, RECIPE_STATE_ALIGN
ret i32 %base
}
define internal i32 @contraction_model_after_state(i32 %elements) #1 {
entry:
%bytes = mul i32 %elements, RECIPE_STATE_ALIGN
%pad = sub i32 RECIPE_MODEL_BYTES, 1
%rounded = add i32 %bytes, %pad
%base = udiv i32 %rounded, RECIPE_MODEL_BYTES
ret i32 %base
}
define internal <RECIPE_REGISTER_M x double> @contraction_a_fragment_vector(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k) #1 {
entry:
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_M x double>, ptr addrspace(3) %ptr, align 8
ret <RECIPE_REGISTER_M x double> %fragment
}
define internal <RECIPE_REGISTER_M x RECIPE_STATE> @contraction_a_fragment_state(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k) #1 {
entry:
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %index
%fragment = load <RECIPE_REGISTER_M x RECIPE_STATE>, ptr addrspace(3) %ptr, align RECIPE_STATE_ALIGN
ret <RECIPE_REGISTER_M x RECIPE_STATE> %fragment
}
define internal <RECIPE_REGISTER_N x RECIPE_STATE> @contraction_b_fragment_state(i32 %k, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%a.elements = mul i32 %tile.m, %tile.k
%base = call i32 @contraction_state_after_model(i32 %a.elements)
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %output.n.base, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %index
%fragment = load <RECIPE_REGISTER_N x RECIPE_STATE>, ptr addrspace(3) %ptr, align RECIPE_STATE_ALIGN
ret <RECIPE_REGISTER_N x RECIPE_STATE> %fragment
}
define internal <RECIPE_REGISTER_N x double> @contraction_b_fragment_after_state(i32 %k, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%a.elements = mul i32 %tile.m, %tile.k
%base = call i32 @contraction_model_after_state(i32 %a.elements)
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %output.n.base, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
%fragment = load <RECIPE_REGISTER_N x double>, ptr addrspace(3) %ptr, align 8
ret <RECIPE_REGISTER_N x double> %fragment
}
define internal void @contraction_stage_b_fragment_state(<RECIPE_FRAGMENT_K x RECIPE_STATE> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
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
%value = extractelement <RECIPE_FRAGMENT_K x RECIPE_STATE> %fragment, i32 %element
%target = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %index
store RECIPE_STATE %value, ptr addrspace(3) %target, align RECIPE_STATE_ALIGN
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_b_fragment_after_state(<RECIPE_FRAGMENT_K x double> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
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
%value = extractelement <RECIPE_FRAGMENT_K x double> %fragment, i32 %element
%target = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %index
store double %value, ptr addrspace(3) %target, align 8
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_delta_a_fragment_state(<RECIPE_FRAGMENT_K x RECIPE_STATE> %delta, <RECIPE_FRAGMENT_K x double> %output, i1 %relu, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, RECIPE_FRAGMENT_K
br i1 %more, label %step, label %exit
step:
%delta.value = extractelement <RECIPE_FRAGMENT_K x RECIPE_STATE> %delta, i32 %element
%output.value = extractelement <RECIPE_FRAGMENT_K x double> %output, i32 %element
%positive = call i1 @recipe.ogt(double %output.value, double 0.0)
%active = select i1 %positive, RECIPE_STATE %delta.value, RECIPE_STATE %zero
%value = select i1 %relu, RECIPE_STATE %active, RECIPE_STATE %delta.value
%local.k = add i32 %k, %element
%index = call i32 @contraction_vector_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
%target = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %index
store RECIPE_STATE %value, ptr addrspace(3) %target, align RECIPE_STATE_ALIGN
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
br i1 %local, label %local.owner.check, label %shared.chunk.loop
local.owner.check:
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
define internal void @contraction_bias_accumulate_state(
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
%a.elements = mul i32 %tile.m, %tile.k %base = call i32 @contraction_state_after_model(i32 %a.elements) %local = call i32 @contraction_vector_b_index(i32 %r, i32 %channel, i32 %tile.n, i32 %tile.k) %index = add i32 %base, %local
%ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %index
%value = load RECIPE_STATE, ptr addrspace(3) %ptr, align RECIPE_STATE_ALIGN %sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %value)
%r.next = add i32 %r, 1 br label %r.loop
sum.store:
store RECIPE_STATE %sum, ptr addrspace(5) %sum.ptr, align RECIPE_STATE_ALIGN br i1 %last, label %destination.store, label %channel.done
destination.store:
%filter = add i32 %n.base, %channel %bias.base = mul i32 %out.channels, %window %bias.local = add i32 %bias.base, %filter %bias.index = add i32 %store.offset, %bias.local
%bias.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %destination, i32 %bias.index
store RECIPE_STATE %sum, ptr addrspace(1) %bias.ptr, align RECIPE_STATE_ALIGN br label %channel.done
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
; The model compiler replaces this with one arm per packed node.
define internal double @recipe.model.decode(ptr addrspace(1) %matrix, i64 %index, i32 %node) #1 { entry: unreachable }
; The dot of one run of consecutive stored weights of a node (a whole block,
; or 32 values of smaller blocks), from %index on, with as many staged
; activations %stride apart: the node's format decodes each value where it lies.
define internal RECIPE_STATE @recipe.model.dot.run(ptr addrspace(1) %matrix, i64 %index, i32 %node, ptr addrspace(3) %x, i32 %stride, i64 %length) #1 { entry: unreachable }
define internal <4 x RECIPE_STATE> @recipe.model.dot.run4(ptr addrspace(1) %matrix, i64 %index, i32 %node, ptr addrspace(3) %x, i32 %stride, i32 %pitch, i64 %length) #1 { entry: unreachable }
; One weight of a node-relative span: a dense node loads it, a packed node decodes it.
define internal double @recipe.model.weight(ptr addrspace(1) %weights, i64 %index, i32 %decode) #1 { entry:
%packed = icmp ne i32 %decode, 0 br i1 %packed, label %decoded, label %direct
direct: %ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %index %loaded = load double, ptr addrspace(1) %ptr, align 8 ret double %loaded
decoded: %value = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %index, i32 %decode) ret double %value }
; Single-position GEMV. Decode appends one position at a time, so one output
; row can be assigned to each lane without staging a full input vector or
; synchronizing every K tile. The wrapper below selects this only for the
; exact one-position, non-activation forward case.
define internal void @contraction_forward_gemv_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %out.begin, i32 %out.span, i32 %kernel,
i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode ) #1 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
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
%sum = phi RECIPE_STATE [ %state.zero, %job.step ], [ %sum.next, %weight.ready ]
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
%weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %weight.local.index
%weight.dense.model = load double, ptr addrspace(1) %weight.ptr, align 8
br label %weight.ready
weight.packed.load:
%weight.packed.model = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %weight.decode.index, i32 %decode)
br label %weight.ready
weight.ready:
%weight.model = phi double [ %weight.dense.model, %weight.dense.load ], [ %weight.packed.model, %weight.packed.load ]
%input.channel = zext i32 %k to i64
%input.length.wide = zext i32 %in.length to i64
%input.offset = mul i64 %input.channel, %input.length.wide
%input.index = add i64 %input.offset, %position
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %input.index
%input.model = load double, ptr addrspace(1) %input.ptr, align 8
%weight.wide = call RECIPE_STATE @recipe.decode(double %weight.model)
%input.wide = call RECIPE_STATE @recipe.decode(double %input.model)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %weight.wide, RECIPE_STATE %input.wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %product)
%k.next = add i32 %k, 1
br label %sum.loop
sum.done:
%bias.base = mul i32 %out.channels, %terms
%bias.index = add i32 %bias.base, %channel
%bias.wide.index = zext i32 %bias.index to i64
br i1 %has.bias, label %bias.load, label %bias.zero
bias.load:
%bias.decode.index = add i64 %weight.base.wide, %bias.wide.index
%bias.packed = icmp ne i32 %decode, 0
br i1 %bias.packed, label %bias.packed.load, label %bias.dense.load
bias.dense.load:
%bias.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %bias.wide.index
%bias.dense.model = load double, ptr addrspace(1) %bias.ptr, align 8
br label %bias.ready
bias.packed.load:
%bias.packed.model = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %bias.decode.index, i32 %decode)
br label %bias.ready
bias.zero:
%bias.zero.model = call double @recipe.encode(RECIPE_STATE %state.zero)
br label %bias.ready
bias.ready:
%bias.model = phi double [ %bias.dense.model, %bias.dense.load ], [ %bias.packed.model, %bias.packed.load ], [ %bias.zero.model, %bias.zero ]
%bias.wide = call RECIPE_STATE @recipe.decode(double %bias.model)
%sum.bias = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %bias.wide)
%sum.value = select i1 %has.bias, RECIPE_STATE %sum.bias, RECIPE_STATE %sum
%result.model = call double @recipe.encode(RECIPE_STATE %sum.value)
%result.positive = call i1 @recipe.ogt(double %result.model, double 0.0)
%result.activated = select i1 %result.positive, double %result.model, double 0.0
%result = select i1 %relu, double %result.activated, double %result.model
%output.channel = zext i32 %channel to i64
%out.length.wide = zext i32 %out.length to i64
%output.channel.base = mul i64 %output.channel, %out.length.wide
%output.index = add i64 %output.channel.base, %position
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.index
store double %result, ptr addrspace(1) %output.ptr, align 8
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
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%width = call i32 @recipe.wavefront.width()
%waves = udiv i32 %block, %width
%wave = udiv i32 %lid, %width
%lane = urem i32 %lid, %width
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
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
%tile.capacity = udiv i32 %tile.bytes, RECIPE_MODEL_BYTES
%tile.blocks = udiv i32 %tile.capacity, 256
%tile.chunk = mul i32 %tile.blocks, 256
%chunk.span = select i1 %stage.available, i32 %tile.chunk, i32 %terms
%stage.tile = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 0
%q8.shared = getelementptr i8, ptr addrspace(3) @contraction_tile, i64 0
%q8.blocks = udiv i32 %terms, 32
; One int8 activation record per 32 inputs: the step's scale in the state,
; then the 32 codes.
%int.code.bytes = select i1 %int16, i64 2, i64 1
%int.block.bytes = mul i64 32, %int.code.bytes
%q8.record = add i64 %int.block.bytes, RECIPE_STATE_ALIGN
%q8.block.record = mul i64 %q8.record, 8
; The inputs that share one step, in 32-value records: the block's int(n)
; step, 32 as the weight blocks or what it declared (256 is llama.cpp's Q8_K).
%q8.step = call i32 @recipe.model.int.step(i32 %decode)
%q8.span = udiv i32 %q8.step, 32
%q8.groups.adjusted = add i32 %q8.blocks, %q8.span
%q8.groups.numerator = sub i32 %q8.groups.adjusted, 1
%q8.groups = udiv i32 %q8.groups.numerator, %q8.span
%q8.width.half = udiv i32 %width, 2
%q8.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%q8.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%int.levels = select i1 %int16, i32 32767, i32 127
%int.minimum = select i1 %int16, i32 -32768, i32 -128
%q8.levels = call RECIPE_STATE @recipe.state.from.u32(i32 %int.levels)
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
%stage.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.at
%stage.value = load double, ptr addrspace(1) %stage.ptr, align 8
%stage.within = and i32 %stage.c, 15
%stage.quad = lshr i32 %stage.within, 2
%stage.lane = and i32 %stage.c, 3
%stage.column = lshr i32 %stage.c, 4
%stage.column.quad = mul i32 %stage.column, 4
%stage.row.base = mul i32 %stage.quad, %chunk.pitch
%stage.at.column = add i32 %stage.row.base, %stage.column.quad
%stage.at.slot = add i32 %stage.at.column, %stage.lane
%stage.slot = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %stage.at.slot
store double %stage.value, ptr addrspace(3) %stage.slot, align 8
%stage.c.next = add i32 %stage.c, %block
br label %stage.loop
stage.done:
call void @recipe.local.barrier()
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
%q8.extreme = phi RECIPE_STATE [ %q8.zero, %q8.group.step ], [ %q8.extreme.acc, %q8.extreme.lane.done ]
%q8.extreme.more = icmp ult i32 %q8.extreme.block, %q8.group.stop
br i1 %q8.extreme.more, label %q8.extreme.step, label %q8.extreme.done
; Each lane walks the block's values at the wave width: one value per lane
; in a wave of 32, every value on a width of one.
q8.extreme.step:
%q8.extreme.term.base = mul i32 %q8.extreme.block, 32
br label %q8.extreme.lane.loop
q8.extreme.lane.loop:
%q8.extreme.v = phi i32 [ %lane, %q8.extreme.step ], [ %q8.extreme.v.next, %q8.extreme.lane.step ]
%q8.extreme.acc = phi RECIPE_STATE [ %q8.extreme, %q8.extreme.step ], [ %q8.extreme.next, %q8.extreme.lane.step ]
%q8.extreme.lane.more = icmp ult i32 %q8.extreme.v, 32
br i1 %q8.extreme.lane.more, label %q8.extreme.lane.step, label %q8.extreme.lane.done
q8.extreme.lane.step:
%q8.extreme.term = add i32 %q8.extreme.term.base, %q8.extreme.v
%q8.extreme.term.wide = zext i32 %q8.extreme.term to i64
%q8.extreme.offset = mul i64 %q8.extreme.term.wide, %in.length.wide
%q8.extreme.index = add i64 %q8.extreme.offset, %position
%q8.extreme.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %q8.extreme.index
%q8.extreme.model = load double, ptr addrspace(1) %q8.extreme.ptr, align 8
%q8.extreme.value = call RECIPE_STATE @recipe.decode(double %q8.extreme.model)
%q8.extreme.abs = call RECIPE_STATE @recipe.state.abs(RECIPE_STATE %q8.extreme.value)
%q8.extreme.have = call RECIPE_STATE @recipe.state.abs(RECIPE_STATE %q8.extreme.acc)
%q8.extreme.larger = call i1 @recipe.state.ogt(RECIPE_STATE %q8.extreme.abs, RECIPE_STATE %q8.extreme.have)
%q8.extreme.next = select i1 %q8.extreme.larger, RECIPE_STATE %q8.extreme.value, RECIPE_STATE %q8.extreme.acc
%q8.extreme.v.next = add i32 %q8.extreme.v, %width
br label %q8.extreme.lane.loop
q8.extreme.lane.done:
%q8.extreme.block.next = add i32 %q8.extreme.block, 1
br label %q8.extreme.loop
q8.extreme.done:
br label %q8.max.loop
q8.max.loop:
%q8.max.offset = phi i32 [ %q8.width.half, %q8.extreme.done ], [ %q8.max.offset.next, %q8.max.step ]
%q8.max.value = phi RECIPE_STATE [ %q8.extreme, %q8.extreme.done ], [ %q8.max.next, %q8.max.step ]
%q8.max.more = icmp ugt i32 %q8.max.offset, 0
br i1 %q8.max.more, label %q8.max.step, label %q8.max.done
q8.max.step:
%q8.max.partner.lane = xor i32 %lane, %q8.max.offset
%q8.max.partner.index = mul i32 %q8.max.partner.lane, 4
%q8.max.partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %q8.max.value, i32 %q8.max.partner.index)
%q8.max.partner.abs = call RECIPE_STATE @recipe.state.abs(RECIPE_STATE %q8.max.partner)
%q8.max.value.abs = call RECIPE_STATE @recipe.state.abs(RECIPE_STATE %q8.max.value)
%q8.max.greater = call i1 @recipe.state.ogt(RECIPE_STATE %q8.max.partner.abs, RECIPE_STATE %q8.max.value.abs)
%q8.max.equal = fcmp oeq RECIPE_STATE %q8.max.partner.abs, %q8.max.value.abs
%q8.max.signed = call i1 @recipe.state.ogt(RECIPE_STATE %q8.max.partner, RECIPE_STATE %q8.max.value)
%q8.max.tie = and i1 %q8.max.equal, %q8.max.signed
%q8.max.choose = or i1 %q8.max.greater, %q8.max.tie
%q8.max.next = select i1 %q8.max.choose, RECIPE_STATE %q8.max.partner, RECIPE_STATE %q8.max.value
%q8.max.offset.next = udiv i32 %q8.max.offset, 2
br label %q8.max.loop
q8.max.done:
%q8.max.abs = call RECIPE_STATE @recipe.state.abs(RECIPE_STATE %q8.max.value)
%q8.nonzero = call i1 @recipe.state.ogt(RECIPE_STATE %q8.max.abs, RECIPE_STATE %q8.zero)
%q8.max.safe = select i1 %q8.nonzero, RECIPE_STATE %q8.max.value, RECIPE_STATE %q8.one
; Match signed scale and reciprocal before rounding codes.
%q8.levels.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %q8.levels)
%q8.inverse.raw = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %q8.levels.negative, RECIPE_STATE %q8.max.safe)
%q8.d.raw = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %q8.one, RECIPE_STATE %q8.inverse.raw)
%q8.inverse = select i1 %q8.nonzero, RECIPE_STATE %q8.inverse.raw, RECIPE_STATE %q8.zero
%q8.d = select i1 %q8.nonzero, RECIPE_STATE %q8.d.raw, RECIPE_STATE %q8.zero
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
store RECIPE_STATE %q8.d, ptr addrspace(3) %q8.code.block.ptr, align RECIPE_STATE_ALIGN
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
%q8.code.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %q8.code.index
%q8.code.model = load double, ptr addrspace(1) %q8.code.ptr, align 8
%q8.code.value = call RECIPE_STATE @recipe.decode(double %q8.code.model)
%q8.code.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %q8.code.value, RECIPE_STATE %q8.inverse)
%q8.code.even = call RECIPE_STATE @recipe.state.roundeven(RECIPE_STATE %q8.code.scaled)
%q8.code.int.raw = call i32 @recipe.state.to.s32(RECIPE_STATE %q8.code.even)
%q8.code.low = icmp slt i32 %q8.code.int.raw, %int.minimum
%q8.code.low.clamped = select i1 %q8.code.low, i32 %int.minimum, i32 %q8.code.int.raw
%q8.code.high = icmp sgt i32 %q8.code.low.clamped, %int.levels
%q8.code.int = select i1 %q8.code.high, i32 %int.levels, i32 %q8.code.low.clamped
br label %q8.code.store
q8.code.store:
%q8.code.v.wide = zext i32 %q8.code.v to i64
%q8.code.ptr.base = getelementptr i8, ptr addrspace(3) %q8.code.block.ptr, i64 RECIPE_STATE_ALIGN
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
call void @recipe.local.barrier()
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
br i1 %exact, label %exact.q4.check, label %q4.check
q4.check:
br i1 %row.q4.block, label %q4b.loop, label %q4.check.slices
q4.check.slices:
br i1 %row.q4.on, label %q4.sum.loop, label %q6.check
q4b.loop:
%q4b.block = phi i32 [ 0, %q4.check ], [ %q4b.block.next, %q4b.done ]
%q4b.acc = phi RECIPE_STATE [ %state.zero, %q4.check ], [ %q4b.acc.next, %q4b.done ]
%q4b.accmin = phi RECIPE_STATE [ %state.zero, %q4.check ], [ %q4b.accmin.next, %q4b.done ]
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
%q4b.d = call RECIPE_STATE @recipe.state.from.f16(half %q4b.d.bits)
%q4b.dmin.ptr = getelementptr i8, ptr addrspace(1) %q4b.d.ptr, i64 2
%q4b.dmin.bits = load half, ptr addrspace(1) %q4b.dmin.ptr, align 2
%q4b.dmin = call RECIPE_STATE @recipe.state.from.f16(half %q4b.dmin.bits)
%q4b.d8 = load RECIPE_STATE, ptr addrspace(3) %q4b.q8, align RECIPE_STATE_ALIGN
%q4b.scale = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %q4b.d, RECIPE_STATE %q4b.d8)
%q4b.dscale = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %q4b.dmin, RECIPE_STATE %q4b.d8)
%q4b.dot0.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.dot0)
%q4b.min0.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.min0)
%q4b.acc.q0 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.acc, RECIPE_STATE %q4b.dot0.value, RECIPE_STATE %q4b.scale)
%q4b.accmin.q0 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.accmin, RECIPE_STATE %q4b.min0.value, RECIPE_STATE %q4b.dscale)
%q4b.dot1.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.dot1)
%q4b.min1.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.min1)
%q4b.acc.q1 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.acc.q0, RECIPE_STATE %q4b.dot1.value, RECIPE_STATE %q4b.scale)
%q4b.accmin.q1 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.accmin.q0, RECIPE_STATE %q4b.min1.value, RECIPE_STATE %q4b.dscale)
%q4b.dot2.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.dot2)
%q4b.min2.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.min2)
%q4b.acc.q2 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.acc.q1, RECIPE_STATE %q4b.dot2.value, RECIPE_STATE %q4b.scale)
%q4b.accmin.q2 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.accmin.q1, RECIPE_STATE %q4b.min2.value, RECIPE_STATE %q4b.dscale)
%q4b.dot3.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.dot3)
%q4b.min3.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.red.min3)
%q4b.acc.q3 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.acc.q2, RECIPE_STATE %q4b.dot3.value, RECIPE_STATE %q4b.scale)
%q4b.accmin.q3 = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.accmin.q2, RECIPE_STATE %q4b.min3.value, RECIPE_STATE %q4b.dscale)
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
%q4b.dot.all.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.dot.all)
%q4b.min.all.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q4b.min.all)
%q4b.acc.whole = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.acc, RECIPE_STATE %q4b.dot.all.value, RECIPE_STATE %q4b.scale)
%q4b.accmin.whole = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q4b.accmin, RECIPE_STATE %q4b.min.all.value, RECIPE_STATE %q4b.dscale)
%q4b.batch.local = sub i32 %position.index, %out.begin
%q4b.batch.rem = and i32 %out.span, 3
%q4b.gemm.end = sub i32 %out.span, %q4b.batch.rem
%q4b.whole.rows = icmp uge i32 %q4b.batch.local, %q4b.gemm.end
%q4b.tail = call i1 @recipe.model.tail(i32 %decode)
%q4b.batch.last = sub i32 %out.span, 1
%q4b.is.last = icmp eq i32 %q4b.batch.local, %q4b.batch.last
%q4b.tail.last = and i1 %q4b.tail, %q4b.is.last
%q4b.whole = or i1 %q4b.whole.rows, %q4b.tail.last
%q4b.acc.next = select i1 %q4b.whole, RECIPE_STATE %q4b.acc.whole, RECIPE_STATE %q4b.acc.q3
%q4b.accmin.next = select i1 %q4b.whole, RECIPE_STATE %q4b.accmin.whole, RECIPE_STATE %q4b.accmin.q3
br label %q4b.done
q4b.done:
%q4b.block.next = add i32 %q4b.block, 1
br label %q4b.loop
q4b.exit:
%q4b.difference = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %q4b.acc, RECIPE_STATE %q4b.accmin)
%q4b.owner = icmp eq i32 %lane, 0
%q4b.owned = select i1 %q4b.owner, RECIPE_STATE %q4b.difference, RECIPE_STATE %state.zero
%q4b.result = select i1 %channel.active, RECIPE_STATE %q4b.owned, RECIPE_STATE %state.zero
br label %sum.done
exact.q4.check:
br i1 %row.q4.on, label %exact.q4.loop, label %exact.q6.check
exact.q4.loop:
%exact.q4.slice = phi i32 [ %lane, %exact.q4.check ], [ %exact.q4.slice.next, %exact.q4.step ]
%exact.q4.sum = phi RECIPE_STATE [ %state.zero, %exact.q4.check ], [ %exact.q4.sum.next, %exact.q4.step ]
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
%exact.q4.loaded = call RECIPE_STATE @recipe.q4k.exact(ptr addrspace(1) %weights, i64 %exact.q4.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.q4.slice, i32 %chunk.pitch, i32 %exact.q4.slice.local)
%exact.q4.value = select i1 %channel.active, RECIPE_STATE %exact.q4.loaded, RECIPE_STATE %state.zero
%exact.q4.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %exact.q4.sum, RECIPE_STATE %exact.q4.value)
%exact.q4.slice.next = add i32 %exact.q4.slice, %width
br label %exact.q4.loop
exact.q4.done:
br label %sum.done
exact.q6.check:
br i1 %row.q6.on, label %exact.q6.loop, label %exact.b32.check
exact.q6.loop:
%exact.q6.slice = phi i32 [ %lane, %exact.q6.check ], [ %exact.q6.slice.next, %exact.q6.step ]
%exact.q6.sum = phi RECIPE_STATE [ %state.zero, %exact.q6.check ], [ %exact.q6.sum.next, %exact.q6.step ]
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
%exact.q6.loaded = call RECIPE_STATE @recipe.q6k.exact(ptr addrspace(1) %weights, i64 %exact.q6.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.q6.slice, i32 %chunk.pitch, i32 %exact.q6.slice.local)
%exact.q6.value = select i1 %channel.active, RECIPE_STATE %exact.q6.loaded, RECIPE_STATE %state.zero
%exact.q6.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %exact.q6.sum, RECIPE_STATE %exact.q6.value)
%exact.q6.slice.next = add i32 %exact.q6.slice, %width
br label %exact.q6.loop
exact.q6.done:
br label %sum.done
exact.b32.check:
br i1 %b32.available, label %exact.b32.loop, label %sum.loop
exact.b32.loop:
%exact.b32.slice = phi i32 [ %lane, %exact.b32.check ], [ %exact.b32.slice.next, %exact.b32.loaded.done ]
%exact.b32.sum = phi RECIPE_STATE [ %state.zero, %exact.b32.check ], [ %exact.b32.sum.next, %exact.b32.loaded.done ]
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
%exact.b32.value32 = call RECIPE_STATE @recipe.block32.fp32(i32 %b32.kind, ptr addrspace(1) %weights, i64 %exact.b32.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.b32.slice, i32 %chunk.pitch)
br label %exact.b32.loaded.done
exact.b32.load16:
%exact.b32.value16 = call RECIPE_STATE @recipe.block32.exact(i32 %b32.kind, ptr addrspace(1) %weights, i64 %exact.b32.byte.offset, ptr addrspace(3) %stage.tile, i32 %exact.b32.slice, i32 %chunk.pitch, i32 %exact.b32.slice.local)
br label %exact.b32.loaded.done
exact.b32.loaded.done:
%exact.b32.loaded = phi RECIPE_STATE [ %exact.b32.value32, %exact.b32.load32 ], [ %exact.b32.value16, %exact.b32.load16 ]
%exact.b32.value = select i1 %channel.active, RECIPE_STATE %exact.b32.loaded, RECIPE_STATE %state.zero
%exact.b32.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %exact.b32.sum, RECIPE_STATE %exact.b32.value)
%exact.b32.slice.next = add i32 %exact.b32.slice, %width
br label %exact.b32.loop
exact.b32.done:
br label %sum.done
q4.sum.loop:
%q4.slice = phi i32 [ %lane, %q4.check.slices ], [ %q4.slice.next, %q4.slice.ready ]
%q4.sum = phi RECIPE_STATE [ %state.zero, %q4.check.slices ], [ %q4.sum.next, %q4.slice.ready ]
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
%q4.loaded = call RECIPE_STATE @recipe.q4k.slice(ptr addrspace(1) %weights, i64 %q4.byte.offset, ptr addrspace(3) %q4.q8.ptr, i32 %q4.slice.local)
%q4.value.active = select i1 %channel.active, RECIPE_STATE %q4.loaded, RECIPE_STATE %state.zero
%q4.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q4.sum, RECIPE_STATE %q4.value.active)
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
%q6b.f0 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f0.next, %q6b.done ]
%q6b.f1 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f1.next, %q6b.done ]
%q6b.f2 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f2.next, %q6b.done ]
%q6b.f3 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f3.next, %q6b.done ]
%q6b.f4 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f4.next, %q6b.done ]
%q6b.f5 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f5.next, %q6b.done ]
%q6b.f6 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f6.next, %q6b.done ]
%q6b.f7 = phi RECIPE_STATE [ %state.zero, %q6.check ], [ %q6b.f7.next, %q6b.done ]
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
%q6b.d = call RECIPE_STATE @recipe.state.from.f16(half %q6b.d.bits)
%q6b.d8 = load RECIPE_STATE, ptr addrspace(3) %q6b.q8, align RECIPE_STATE_ALIGN
%q6b.scale = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %q6b.d8, RECIPE_STATE %q6b.d)
%q6b.i0.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i0)
%q6b.f0.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f0, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i0.value)
%q6b.i1.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i1)
%q6b.f1.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f1, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i1.value)
%q6b.i2.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i2)
%q6b.f2.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f2, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i2.value)
%q6b.i3.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i3)
%q6b.f3.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f3, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i3.value)
%q6b.i4.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i4)
%q6b.f4.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f4, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i4.value)
%q6b.i5.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i5)
%q6b.f5.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f5, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i5.value)
%q6b.i6.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i6)
%q6b.f6.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f6, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i6.value)
%q6b.i7.value = call RECIPE_STATE @recipe.state.from.s32(i32 %q6b.red.i7)
%q6b.f7.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %q6b.f7, RECIPE_STATE %q6b.scale, RECIPE_STATE %q6b.i7.value)
br label %q6b.done
q6b.done:
%q6b.block.next = add i32 %q6b.block, 1
br label %q6b.loop
q6b.exit:
%q6b.t0 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6b.f0, RECIPE_STATE %q6b.f4)
%q6b.t1 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6b.f1, RECIPE_STATE %q6b.f5)
%q6b.t2 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6b.f2, RECIPE_STATE %q6b.f6)
%q6b.t3 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6b.f3, RECIPE_STATE %q6b.f7)
%q6b.u0 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6b.t0, RECIPE_STATE %q6b.t2)
%q6b.u1 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6b.t1, RECIPE_STATE %q6b.t3)
%q6b.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6b.u0, RECIPE_STATE %q6b.u1)
%q6b.owner = icmp eq i32 %lane, 0
%q6b.owned = select i1 %q6b.owner, RECIPE_STATE %q6b.total, RECIPE_STATE %state.zero
%q6b.result = select i1 %channel.active, RECIPE_STATE %q6b.owned, RECIPE_STATE %state.zero
br label %sum.done
q6.sum.loop:
%q6.slice = phi i32 [ %lane, %q6.check.slices ], [ %q6.slice.next, %q6.slice.ready ]
%q6.sum = phi RECIPE_STATE [ %state.zero, %q6.check.slices ], [ %q6.sum.next, %q6.slice.ready ]
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
%q6.loaded = call RECIPE_STATE @recipe.q6k.slice(ptr addrspace(1) %weights, i64 %q6.byte.offset, ptr addrspace(3) %q6.q8.ptr, i32 %q6.slice.local)
%q6.value.active = select i1 %channel.active, RECIPE_STATE %q6.loaded, RECIPE_STATE %state.zero
%q6.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %q6.sum, RECIPE_STATE %q6.value.active)
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
%b32.sum = phi RECIPE_STATE [ %state.zero, %b32.check ], [ %b32.sum.next, %b32.slice.ready ]
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
%b32.value16 = call RECIPE_STATE @recipe.block32.i16(i32 %b32.kind, ptr addrspace(1) %weights, i64 %b32.byte.offset, ptr addrspace(3) %b32.q8.ptr, i32 %b32.slice.local)
br label %b32.loaded.done
b32.load8:
%b32.value8 = call RECIPE_STATE @recipe.block32.slice(i32 %b32.kind, ptr addrspace(1) %weights, i64 %b32.byte.offset, ptr addrspace(3) %b32.q8.ptr, i32 %b32.slice.local)
br label %b32.loaded.done
b32.loaded.done:
%b32.loaded = phi RECIPE_STATE [ %b32.value16, %b32.load16 ], [ %b32.value8, %b32.load8 ]
%b32.value.active = select i1 %channel.active, RECIPE_STATE %b32.loaded, RECIPE_STATE %state.zero
%b32.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %b32.sum, RECIPE_STATE %b32.value.active)
%b32.slice.next = add i32 %b32.slice, %width
br label %b32.slice.ready
b32.slice.ready:
br label %b32.sum.loop
b32.sum.done:
br label %sum.done
sum.loop:
%k = phi i32 [ %lane, %b32.check ], [ %lane, %exact.b32.check ], [ %k.next, %weight.ready ]
%sum = phi RECIPE_STATE [ %state.zero, %b32.check ], [ %state.zero, %exact.b32.check ], [ %sum.next, %weight.ready ]
%k.more = icmp ult i32 %k, %terms
br i1 %k.more, label %sum.step, label %sum.done
sum.step:
%k.wide = zext i32 %k to i64
%weight.local.index = add i64 %channel.offset, %k.wide
%weight.decode.index = add i64 %weight.base.wide, %weight.local.index
%weight.packed = icmp ne i32 %decode, 0
br i1 %weight.packed, label %weight.packed.load, label %weight.dense.load
weight.dense.load:
%weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %weight.local.index
%weight.dense.model = load double, ptr addrspace(1) %weight.ptr, align 8
br label %weight.ready
weight.packed.load:
%weight.packed.model = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %weight.decode.index, i32 %decode)
br label %weight.ready
weight.ready:
%weight.model = phi double [ %weight.dense.model, %weight.dense.load ], [ %weight.packed.model, %weight.packed.load ]
%input.channel = zext i32 %k to i64
%input.offset = mul i64 %input.channel, %in.length.wide
%input.index = add i64 %input.offset, %position
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %input.index
%input.model = load double, ptr addrspace(1) %input.ptr, align 8
%weight.wide = call RECIPE_STATE @recipe.decode(double %weight.model)
%input.wide = call RECIPE_STATE @recipe.decode(double %input.model)
%product.raw = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %weight.wide, RECIPE_STATE %input.wide)
%product = select i1 %channel.active, RECIPE_STATE %product.raw, RECIPE_STATE %state.zero
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %product)
%k.next = add i32 %k, %width
br label %sum.loop
sum.done:
%sum.final = phi RECIPE_STATE [ %sum, %sum.loop ], [ %q4b.result, %q4b.exit ], [ %q4.sum, %q4.sum.done ], [ %q6.sum, %q6.sum.done ], [ %q6b.result, %q6b.exit ], [ %b32.sum, %b32.sum.done ], [ %exact.q4.sum, %exact.q4.done ], [ %exact.q6.sum, %exact.q6.done ], [ %exact.b32.sum, %exact.b32.done ]
%reduce.offset.initial = udiv i32 %width, 2
br label %reduce.loop
reduce.loop:
%reduce.offset = phi i32 [ %reduce.offset.initial, %sum.done ], [ %reduce.offset.next, %reduce.step ]
%reduced = phi RECIPE_STATE [ %sum.final, %sum.done ], [ %reduced.next, %reduce.step ]
%reduce.more = icmp ugt i32 %reduce.offset, 0
br i1 %reduce.more, label %reduce.step, label %reduce.done
reduce.step:
%partner.lane = xor i32 %lane, %reduce.offset
%partner.index = mul i32 %partner.lane, 4
%partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %reduced, i32 %partner.index)
%reduced.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %reduced, RECIPE_STATE %partner)
%reduce.offset.next = udiv i32 %reduce.offset, 2
br label %reduce.loop
reduce.done:
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
%bias.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %bias.wide.index
%bias.dense.model = load double, ptr addrspace(1) %bias.ptr, align 8
br label %bias.ready
bias.packed.load:
%bias.packed.model = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %bias.decode.index, i32 %decode)
br label %bias.ready
bias.zero:
%bias.zero.model = call double @recipe.encode(RECIPE_STATE %state.zero)
br label %bias.ready
bias.ready:
%bias.model = phi double [ %bias.dense.model, %bias.dense.load ], [ %bias.packed.model, %bias.packed.load ], [ %bias.zero.model, %bias.zero ]
%bias.wide = call RECIPE_STATE @recipe.decode(double %bias.model)
%sum.bias = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %reduced, RECIPE_STATE %bias.wide)
%bias.now = and i1 %has.bias, %chunk.first
%sum.value = select i1 %bias.now, RECIPE_STATE %sum.bias, RECIPE_STATE %reduced
%output.channel = zext i32 %channel to i64
%out.length.wide = zext i32 %out.length to i64
%output.channel.base = mul i64 %output.channel, %out.length.wide
%output.index = add i64 %output.channel.base, %position
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.index
; A later chunk adds the partial the earlier chunks left in the output; the
; first chunk reads nothing there.
br i1 %chunk.first, label %prior.none, label %prior.load
prior.load:
%prior.model = load double, ptr addrspace(1) %output.ptr, align 8
%prior.wide = call RECIPE_STATE @recipe.decode(double %prior.model)
%sum.carried = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.value, RECIPE_STATE %prior.wide)
br label %prior.none
prior.none:
%sum.total = phi RECIPE_STATE [ %sum.value, %bias.ready ], [ %sum.carried, %prior.load ]
%result.model = call double @recipe.encode(RECIPE_STATE %sum.total)
%result.positive = call i1 @recipe.ogt(double %result.model, double 0.0)
%result.activated = select i1 %result.positive, double %result.model, double 0.0
%relu.now = and i1 %relu, %chunk.last
%result = select i1 %relu.now, double %result.activated, double %result.model
store double %result, ptr addrspace(1) %output.ptr, align 8
br label %job.done
job.done:
%job.next = add i32 %job, %groups
br label %job.loop
chunk.done:
call void @recipe.local.barrier()
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
%block = call i32 @recipe.workgroup.size.x()
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
; The part of one unit (a block, or a run of smaller blocks) of row %row of a
; packed sum: the run dot of every place in the unit against the staged column,
; scaled by the slot's routing weight for an expert's down table; zero when the
; unit is inactive.
define internal RECIPE_STATE @packed_unit_dot( ptr addrspace(1) %weights, i32 %node, i1 %expert, i1 %down, i32 %hidden.nonzero, i64 %hidden.wide, i64 %rows.wide, i64 %terms.wide, i64 %weight.base,
i32 %chunk.base, i64 %chunk.base.wide, i32 %unit, i32 %cases, i32 %upr, ptr addrspace(3) %ids.base, ptr addrspace(3) %scales.base, i32 %row, i32 %unit.index, i1 %active ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
; The row length of the stored table: a down table stores rows of the hidden width.
%row.length = select i1 %down, i64 %hidden.wide, i64 %terms.wide
%row.safe = select i1 %active, i32 %row, i32 0
%unit.safe = select i1 %active, i32 %unit.index, i32 0
%unit.start = mul i32 %unit.safe, %unit
%k0 = add i32 %chunk.base, %unit.start
; The weight row this unit reads, as the index of its first value less k0.
%in.slot = udiv i32 %row.safe, %hidden.nonzero
%down.slot = udiv i32 %k0, %hidden.nonzero
%slot.any = select i1 %down, i32 %down.slot, i32 %in.slot
%slot = select i1 %expert, i32 %slot.any, i32 0
%slot.id.ptr = getelementptr i32, ptr addrspace(3) %ids.base, i32 %slot
%slot.id = load i32, ptr addrspace(3) %slot.id.ptr, align 4
%slot.scale.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %scales.base, i32 %slot
%slot.scale = load RECIPE_STATE, ptr addrspace(3) %slot.scale.ptr, align RECIPE_STATE_ALIGN
%slot.id.wide = zext i32 %slot.id to i64
%row.wide = zext i32 %row.safe to i64
%in.f = urem i32 %row.safe, %hidden.nonzero
%in.f.wide = zext i32 %in.f to i64
%in.erow.base = mul i64 %slot.id.wide, %hidden.wide
%in.erow = add i64 %in.erow.base, %in.f.wide
%in.row.index = mul i64 %in.erow, %terms.wide
%down.erow.base = mul i64 %slot.id.wide, %rows.wide
%down.erow = add i64 %down.erow.base, %row.wide
%down.row.index = mul i64 %down.erow, %hidden.wide
%down.slot.wide = zext i32 %slot to i64
%down.slot.start = mul i64 %down.slot.wide, %hidden.wide
%down.row.shifted = sub i64 %down.row.index, %down.slot.start
%plain.row.index = mul i64 %row.wide, %terms.wide
%expert.row.index = select i1 %down, i64 %down.row.shifted, i64 %in.row.index
%row.index = select i1 %expert, i64 %expert.row.index, i64 %plain.row.index
%row.base = add i64 %weight.base, %row.index
%k0.wide = zext i32 %k0 to i64
%run.first = add i64 %row.base, %k0.wide
br label %case.loop
case.loop:
%case = phi i32 [ 0, %entry ], [ %case.next, %case.step ]
%part = phi RECIPE_STATE [ %state.zero, %entry ], [ %part.next, %case.step ]
%case.more = icmp ult i32 %case, %cases
br i1 %case.more, label %case.step, label %case.done
case.step:
%case.values = mul i32 %case, %unit
%case.wide = zext i32 %case.values to i64
%run.index = add i64 %run.first, %case.wide
%x.pitch = add i32 %unit, 1
%x.row = mul i32 %unit.safe, %x.pitch
%x.at = add i32 %x.row, %case.values
%x = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %x.at
%value = call RECIPE_STATE @recipe.model.dot.run(ptr addrspace(1) %weights, i64 %run.index, i32 %node, ptr addrspace(3) %x, i32 1, i64 %row.length)
%part.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %part, RECIPE_STATE %value)
%case.next = add i32 %case, 1
br label %case.loop
case.done:
%part.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %part, RECIPE_STATE %slot.scale)
%part.add = select i1 %down, RECIPE_STATE %part.scaled, RECIPE_STATE %part
%part.active = select i1 %active, RECIPE_STATE %part.add, RECIPE_STATE %state.zero
ret RECIPE_STATE %part.active
}
; Row %out.row of a packed sum at %position: the sum plus the bias on the first
; chunk, plus what earlier chunks left in the output, rectified on the last chunk.
define internal void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last,
i32 %out.row, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position, i64 %weight.base, RECIPE_STATE %sum, i32 %row.first, i32 %row.period, i32 %row.share ) #1 { entry:
%row.share.zero = icmp eq i32 %row.share, 0
%row.share.nonzero = select i1 %row.share.zero, i32 1, i32 %row.share
%bias.row = mul i32 %rows, %terms
%bias.at = add i32 %bias.row, %out.row
%bias.wide = zext i32 %bias.at to i64
%bias.index = add i64 %weight.base, %bias.wide
%bias.now = and i1 %has.bias, %chunk.first
br i1 %bias.now, label %bias, label %biased
bias:
%bias.model = call double @recipe.model.weight(ptr addrspace(1) %weights, i64 %bias.index, i32 %decode)
%bias.state = call RECIPE_STATE @recipe.decode(double %bias.model)
%sum.bias = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %bias.state)
br label %biased
biased:
%sum.b = phi RECIPE_STATE [ %sum, %entry ], [ %sum.bias, %bias ]
; A die of a tensor split holds a share of the rows and writes each at its
; place: rows from %row.first on, or with a %row.period the %row.share rows
; from %row.first on of every period (a share of each expert's rows).
%out.period.index = udiv i32 %out.row, %row.share.nonzero
%out.period.within = urem i32 %out.row, %row.share.nonzero
%out.period.base = mul i32 %out.period.index, %row.period
%out.period.row = add i32 %out.period.base, %out.period.within
%out.periodic = icmp ne i32 %row.period, 0
%out.row.share = select i1 %out.periodic, i32 %out.period.row, i32 %out.row
%out.row.global = add i32 %out.row.share, %row.first
%out.channel = zext i32 %out.row.global to i64
%out.base = mul i64 %out.channel, %out.length.wide
%out.index = add i64 %out.base, %position
%out.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %out.index
br i1 %chunk.first, label %store, label %carry
carry:
%prior.model = load double, ptr addrspace(1) %out.ptr, align 8
%prior = call RECIPE_STATE @recipe.decode(double %prior.model)
%sum.carried = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.b, RECIPE_STATE %prior)
br label %store
store:
%total = phi RECIPE_STATE [ %sum.b, %biased ], [ %sum.carried, %carry ]
%result.model = call double @recipe.encode(RECIPE_STATE %total)
%positive = call i1 @recipe.ogt(double %result.model, double 0.0)
%activated = select i1 %positive, double %result.model, double 0.0
%relu.now = and i1 %relu, %chunk.last
%result = select i1 %relu.now, double %activated, double %result.model
store double %result, ptr addrspace(1) %out.ptr, align 8
ret void
}
; Rows of a plain sum over stored weights at tiles of four positions from
; %out.begin, one row per lane as in the lane body: each decoded weight adds
; into the lane's four positions, the last tile's past the window left unused.
; It takes a window of two positions or more whole, and none when the window
; is one position or the tile cannot hold a unit for four positions.
define internal i32 @packed_lanes4_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %rows, i32 %terms, i32 %in.length, i32 %out.length, i32 %out.begin, i32 %out.span,
i1 %has.bias, i1 %relu, i32 %threads, i64 %weight.base, i32 %decode, i32 %node, i32 %row.first, i32 %row.period, i32 %row.share, i32 %in.first, i32 %in.period, i32 %in.share ) RECIPE_CONTRACTION_BODY { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%width = call i32 @recipe.wavefront.width()
%waves = udiv i32 %block, %width
%wave = udiv i32 %lid, %width
%lane = urem i32 %lid, %width
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%terms.wide = zext i32 %terms to i64
%in.length.wide = zext i32 %in.length to i64
%out.length.wide = zext i32 %out.length to i64
%in.share.zero = icmp eq i32 %in.share, 0
%in.share.nonzero = select i1 %in.share.zero, i32 1, i32 %in.share
%unit = call i32 @recipe.model.dot.run.length(i32 %node)
%unit.wide = zext i32 %unit to i64
; The tile holds four staged columns, then four parts per lane of the workgroup.
%tile.bytes = call i32 @recipe.tile.bytes()
%parts.lane = mul i32 %block, 4
%parts.room = mul i32 %parts.lane, RECIPE_STATE_ALIGN
%fixed.room = add i32 %parts.room, 32
%tile.fits = icmp ugt i32 %tile.bytes, %fixed.room
%tile.left = sub i32 %tile.bytes, %fixed.room
%tile.spare = select i1 %tile.fits, i32 %tile.left, i32 0
%tile.values = udiv i32 %tile.spare, RECIPE_MODEL_BYTES
%unit.four = mul i32 %unit, 4
%chunk.units.room = udiv i32 %tile.values, %unit.four
%units = udiv i32 %terms, %unit
%chunk.units.over = icmp ugt i32 %chunk.units.room, %units
%chunk.units = select i1 %chunk.units.over, i32 %units, i32 %chunk.units.room
%chunk.span = mul i32 %chunk.units, %unit
%x.values = mul i32 %chunk.span, 4
%x.bytes = mul i32 %x.values, RECIPE_MODEL_BYTES
%x.over = add i32 %x.bytes, 15
%x.aligned = and i32 %x.over, -16
%parts = getelementptr i8, ptr addrspace(3) @contraction_tile, i32 %x.aligned
%row.groups.over = add i32 %rows, %width
%row.groups.raised = sub i32 %row.groups.over, 1
%row.groups = udiv i32 %row.groups.raised, %width
%per.group.over = add i32 %row.groups, %groups
%per.group.raised = sub i32 %per.group.over, 1
%per.group.raw = udiv i32 %per.group.raised, %groups
%per.group.some = icmp ugt i32 %per.group.raw, 0
%per.group = select i1 %per.group.some, i32 %per.group.raw, i32 1
%teams.over = icmp ugt i32 %per.group, %waves
%teams = select i1 %teams.over, i32 %waves, i32 %per.group
%team.size = udiv i32 %waves, %teams
%team = udiv i32 %wave, %team.size
%part = urem i32 %wave, %team.size
%team.in = icmp ult i32 %team, %teams
%rounds.over = add i32 %per.group, %teams
%rounds.raised = sub i32 %rounds.over, 1
%rounds = udiv i32 %rounds.raised, %teams
%leads = icmp eq i32 %part, 0
%chunk.some = icmp ugt i32 %chunk.units, 0
; A window of two or more positions takes tiles of four, the last one partial.
%tiles.over = add i32 %out.span, 3
%tiles.raw = udiv i32 %tiles.over, 4
%tiles.span = icmp uge i32 %out.span, 2
%tiles.take = and i1 %chunk.some, %tiles.span
%tiles = select i1 %tiles.take, i32 %tiles.raw, i32 0
%lane.parts = mul i32 %lid, 4
br label %tile.loop
tile.loop:
%tile = phi i32 [ 0, %entry ], [ %tile.next, %chunk.done ]
%tile.more = icmp ult i32 %tile, %tiles
br i1 %tile.more, label %tile.step, label %exit
tile.step:
%tile.first = mul i32 %tile, 4
%tile.remaining = sub i32 %out.span, %tile.first
%tile.whole = icmp ugt i32 %tile.remaining, 4
%tile.count = select i1 %tile.whole, i32 4, i32 %tile.remaining
%position0.index = add i32 %out.begin, %tile.first
%position0 = zext i32 %position0.index to i64
br label %chunk.loop
chunk.loop:
%chunk.base = phi i32 [ 0, %tile.step ], [ %chunk.next, %round.loop.end ]
%chunk.remaining = sub i32 %terms, %chunk.base
%chunk.over = icmp ugt i32 %chunk.remaining, %chunk.span
%chunk.terms = select i1 %chunk.over, i32 %chunk.span, i32 %chunk.remaining
%chunk.end = add i32 %chunk.base, %chunk.terms
%chunk.first = icmp eq i32 %chunk.base, 0
%chunk.last = icmp eq i32 %chunk.end, %terms
%chunk.units.here = udiv i32 %chunk.terms, %unit
%chunk.base.wide = zext i32 %chunk.base to i64
%stage.count = mul i32 %chunk.terms, 4
br label %stage.loop
stage.loop:
%stage.c = phi i32 [ %lid, %chunk.loop ], [ %stage.c.next, %stage.step ]
%stage.more = icmp ult i32 %stage.c, %stage.count
br i1 %stage.more, label %stage.step, label %stage.done
stage.step:
%stage.t = udiv i32 %stage.c, %chunk.terms
%stage.local = urem i32 %stage.c, %chunk.terms
%stage.k = add i32 %stage.local, %chunk.base
%stage.k.period = udiv i32 %stage.k, %in.share.nonzero
%stage.k.within = urem i32 %stage.k, %in.share.nonzero
%stage.k.base = mul i32 %stage.k.period, %in.period
%stage.k.placed = add i32 %stage.k.base, %stage.k.within
%stage.k.periodic = icmp ne i32 %in.period, 0
%stage.k.share = select i1 %stage.k.periodic, i32 %stage.k.placed, i32 %stage.k
%stage.k.global = add i32 %stage.k.share, %in.first
%stage.channel = zext i32 %stage.k.global to i64
%stage.index = mul i64 %stage.channel, %in.length.wide
%stage.t.wide = zext i32 %stage.t to i64
%stage.position = add i64 %position0, %stage.t.wide
%stage.at = add i64 %stage.index, %stage.position
%stage.in = icmp ult i32 %stage.t, %tile.count
%stage.at.safe = select i1 %stage.in, i64 %stage.at, i64 %stage.index
%stage.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.at.safe
%stage.loaded = load double, ptr addrspace(1) %stage.ptr, align 8
%stage.value = select i1 %stage.in, double %stage.loaded, double 0.0
%stage.column = mul i32 %stage.t, %chunk.span
%stage.place = add i32 %stage.column, %stage.local
%stage.slot = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %stage.place
store double %stage.value, ptr addrspace(3) %stage.slot, align 8
%stage.c.next = add i32 %stage.c, %block
br label %stage.loop
stage.done:
call void @recipe.local.barrier()
br label %round.loop
round.loop:
%round = phi i32 [ 0, %stage.done ], [ %round.next, %round.done ]
%round.more = icmp ult i32 %round, %rounds
br i1 %round.more, label %round.step, label %round.loop.end
round.step:
%slot.base = mul i32 %round, %teams
%slot = add i32 %slot.base, %team
%slot.stride = mul i32 %slot, %groups
%row.group = add i32 %slot.stride, %group
%row.group.in = icmp ult i32 %row.group, %row.groups
%team.live = and i1 %team.in, %row.group.in
%row.lane0 = mul i32 %row.group, %width
%row = add i32 %row.lane0, %lane
%row.in = icmp ult i32 %row, %rows
%row.live = and i1 %team.live, %row.in
%row.safe = select i1 %row.live, i32 %row, i32 0
%row.wide = zext i32 %row.safe to i64
%row.index = mul i64 %row.wide, %terms.wide
%row.base = add i64 %weight.base, %row.index
%row.chunk = add i64 %row.base, %chunk.base.wide
br label %unit.loop
unit.loop:
%u = phi i32 [ %part, %round.step ], [ %u.next, %unit.step ]
%sum0 = phi RECIPE_STATE [ %state.zero, %round.step ], [ %sum0.next, %unit.step ]
%sum1 = phi RECIPE_STATE [ %state.zero, %round.step ], [ %sum1.next, %unit.step ]
%sum2 = phi RECIPE_STATE [ %state.zero, %round.step ], [ %sum2.next, %unit.step ]
%sum3 = phi RECIPE_STATE [ %state.zero, %round.step ], [ %sum3.next, %unit.step ]
%u.more = icmp ult i32 %u, %chunk.units.here
%u.go = and i1 %u.more, %team.live
br i1 %u.go, label %unit.step, label %unit.done
unit.step:
%u.start = mul i32 %u, %unit
%u.start.wide = zext i32 %u.start to i64
%run.index = add i64 %row.chunk, %u.start.wide
%x = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %u.start
%runs = call <4 x RECIPE_STATE> @recipe.model.dot.run4(ptr addrspace(1) %weights, i64 %run.index, i32 %node, ptr addrspace(3) %x, i32 1, i32 %chunk.span, i64 %terms.wide)
%run0 = extractelement <4 x RECIPE_STATE> %runs, i32 0
%run1 = extractelement <4 x RECIPE_STATE> %runs, i32 1
%run2 = extractelement <4 x RECIPE_STATE> %runs, i32 2
%run3 = extractelement <4 x RECIPE_STATE> %runs, i32 3
%sum0.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum0, RECIPE_STATE %run0)
%sum1.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum1, RECIPE_STATE %run1)
%sum2.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum2, RECIPE_STATE %run2)
%sum3.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum3, RECIPE_STATE %run3)
%u.next = add i32 %u, %team.size
br label %unit.loop
unit.done:
%part0.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %parts, i32 %lane.parts
store RECIPE_STATE %sum0, ptr addrspace(3) %part0.ptr, align RECIPE_STATE_ALIGN
%part1.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %part0.ptr, i32 1
store RECIPE_STATE %sum1, ptr addrspace(3) %part1.ptr, align RECIPE_STATE_ALIGN
%part2.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %part0.ptr, i32 2
store RECIPE_STATE %sum2, ptr addrspace(3) %part2.ptr, align RECIPE_STATE_ALIGN
%part3.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %part0.ptr, i32 3
store RECIPE_STATE %sum3, ptr addrspace(3) %part3.ptr, align RECIPE_STATE_ALIGN
call void @recipe.local.barrier()
%store.now = and i1 %leads, %row.live
br i1 %store.now, label %gather.loop, label %round.done
gather.loop:
%g = phi i32 [ 1, %unit.done ], [ %g.next, %gather.step ]
%total0 = phi RECIPE_STATE [ %sum0, %unit.done ], [ %total0.next, %gather.step ]
%total1 = phi RECIPE_STATE [ %sum1, %unit.done ], [ %total1.next, %gather.step ]
%total2 = phi RECIPE_STATE [ %sum2, %unit.done ], [ %total2.next, %gather.step ]
%total3 = phi RECIPE_STATE [ %sum3, %unit.done ], [ %total3.next, %gather.step ]
%g.more = icmp ult i32 %g, %team.size
br i1 %g.more, label %gather.step, label %gather.store
gather.step:
%g.wave = add i32 %wave, %g
%g.lane0 = mul i32 %g.wave, %width
%g.lid = add i32 %g.lane0, %lane
%g.slot = mul i32 %g.lid, 4
%g0.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %parts, i32 %g.slot
%g0 = load RECIPE_STATE, ptr addrspace(3) %g0.ptr, align RECIPE_STATE_ALIGN
%g1.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %g0.ptr, i32 1
%g1 = load RECIPE_STATE, ptr addrspace(3) %g1.ptr, align RECIPE_STATE_ALIGN
%g2.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %g0.ptr, i32 2
%g2 = load RECIPE_STATE, ptr addrspace(3) %g2.ptr, align RECIPE_STATE_ALIGN
%g3.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %g0.ptr, i32 3
%g3 = load RECIPE_STATE, ptr addrspace(3) %g3.ptr, align RECIPE_STATE_ALIGN
%total0.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total0, RECIPE_STATE %g0)
%total1.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total1, RECIPE_STATE %g1)
%total2.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total2, RECIPE_STATE %g2)
%total3.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total3, RECIPE_STATE %g3)
%g.next = add i32 %g, 1
br label %gather.loop
gather.store:
%position1 = add i64 %position0, 1
%position2 = add i64 %position0, 2
%position3 = add i64 %position0, 3
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last, i32 %row, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position0, i64 %weight.base, RECIPE_STATE %total0, i32 %row.first, i32 %row.period, i32 %row.share )
%store1 = icmp ugt i32 %tile.count, 1
br i1 %store1, label %gather.store1, label %round.done
gather.store1:
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last, i32 %row, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position1, i64 %weight.base, RECIPE_STATE %total1, i32 %row.first, i32 %row.period, i32 %row.share )
%store2 = icmp ugt i32 %tile.count, 2
br i1 %store2, label %gather.store2, label %round.done
gather.store2:
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last, i32 %row, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position2, i64 %weight.base, RECIPE_STATE %total2, i32 %row.first, i32 %row.period, i32 %row.share )
%store3 = icmp ugt i32 %tile.count, 3
br i1 %store3, label %gather.store3, label %round.done
gather.store3:
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last, i32 %row, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position3, i64 %weight.base, RECIPE_STATE %total3, i32 %row.first, i32 %row.period, i32 %row.share )
br label %round.done
round.done:
call void @recipe.local.barrier()
%round.next = add i32 %round, 1
br label %round.loop
round.loop.end:
%chunk.next = add i32 %chunk.base, %chunk.terms
%chunk.more = icmp ult i32 %chunk.next, %terms
br i1 %chunk.more, label %chunk.loop, label %chunk.done
chunk.done:
%tile.next = add i32 %tile, 1
br label %tile.loop
exit:
%tiles.none = icmp eq i32 %tiles, 0
%handled = select i1 %tiles.none, i32 0, i32 %out.span
ret i32 %handled
}
; Rows of a sum over stored weights at the positions %out.begin to
; %out.begin + %out.span, one row per lane: the lanes of a wave take adjacent
; rows and decode the same unit of their rows at a time, reading one staged
; input value together. A team of waves takes a group of rows and splits their
; units; the team's first wave adds the parts and stores the rows. The modes
; are the row body's: a plain sum, an expert's gate or up table, and an
; expert's down table. It returns the positions it took: none when the tile
; cannot hold a unit, a position routes to more than 64 slots, or a unit of a
; down table would reach into the next slot.
define internal i32 @packed_lanes_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %routing, i32 %rows, i32 %terms, i32 %in.length, i32 %out.length, i32 %out.begin, i32 %out.span,
i1 %has.bias, i1 %relu, i32 %threads, i64 %weight.base, i32 %decode, i32 %node, i32 %mode, i32 %hidden, i32 %experts, i32 %top,
ptr addrspace(1) %split.scratch, i32 %row.first, i32 %row.period, i32 %row.share, i32 %in.first, i32 %in.period, i32 %in.share ) RECIPE_CONTRACTION_BODY { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%width = call i32 @recipe.wavefront.width()
%waves = udiv i32 %block, %width
%wave = udiv i32 %lid, %width
%lane = urem i32 %lid, %width
%wave.first = icmp eq i32 %wave, 0
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%terms.wide = zext i32 %terms to i64
%rows.wide = zext i32 %rows to i64
%in.length.wide = zext i32 %in.length to i64
%out.length.wide = zext i32 %out.length to i64
%in.share.zero = icmp eq i32 %in.share, 0
%in.share.nonzero = select i1 %in.share.zero, i32 1, i32 %in.share
%hidden.zero = icmp eq i32 %hidden, 0
%hidden.nonzero = select i1 %hidden.zero, i32 1, i32 %hidden
%hidden.wide = zext i32 %hidden.nonzero to i64
%expert = icmp ne i32 %mode, 0
%down = icmp eq i32 %mode, 2
%unit = call i32 @recipe.model.dot.run.length(i32 %node)
%unit.wide = zext i32 %unit to i64
; The tile holds the staged input, one part per lane of the workgroup, then
; 64 slot experts and their 64 routing weights.
%tile.bytes = call i32 @recipe.tile.bytes()
%parts.room = mul i32 %block, RECIPE_STATE_ALIGN
%routes.room = mul i32 64, RECIPE_STATE_ALIGN
%routes.room.ids = add i32 %routes.room, 256
%fixed.room.parts = add i32 %parts.room, %routes.room.ids
%fixed.room = add i32 %fixed.room.parts, 32
%tile.fits = icmp ugt i32 %tile.bytes, %fixed.room
%tile.left = sub i32 %tile.bytes, %fixed.room
%tile.spare = select i1 %tile.fits, i32 %tile.left, i32 0
%tile.values = udiv i32 %tile.spare, RECIPE_MODEL_BYTES
%chunk.units.room = udiv i32 %tile.values, %unit
%units = udiv i32 %terms, %unit
%chunk.units.over = icmp ugt i32 %chunk.units.room, %units
%chunk.units = select i1 %chunk.units.over, i32 %units, i32 %chunk.units.room
%chunk.span = mul i32 %chunk.units, %unit
%x.bytes = mul i32 %chunk.span, RECIPE_MODEL_BYTES
%x.over = add i32 %x.bytes, 15
%x.aligned = and i32 %x.over, -16
%parts = getelementptr i8, ptr addrspace(3) @contraction_tile, i32 %x.aligned
%ids = getelementptr i8, ptr addrspace(3) %parts, i32 %parts.room
%scales = getelementptr i8, ptr addrspace(3) %ids, i32 256
; Each workgroup takes every %groups-th group of rows; its teams take them in
; rounds, and with fewer groups than waves a team is several waves.
%row.groups.over = add i32 %rows, %width
%row.groups.raised = sub i32 %row.groups.over, 1
%row.groups = udiv i32 %row.groups.raised, %width
%per.group.over = add i32 %row.groups, %groups
%per.group.raised = sub i32 %per.group.over, 1
%per.group.raw = udiv i32 %per.group.raised, %groups
%per.group.some = icmp ugt i32 %per.group.raw, 0
%per.group = select i1 %per.group.some, i32 %per.group.raw, i32 1
%teams.over = icmp ugt i32 %per.group, %waves
%teams = select i1 %teams.over, i32 %waves, i32 %per.group
%team.size = udiv i32 %waves, %teams
%team = udiv i32 %wave, %team.size
%part = urem i32 %wave, %team.size
%team.in = icmp ult i32 %team, %teams
%rounds.over = add i32 %per.group, %teams
%rounds.raised = sub i32 %rounds.over, 1
%rounds = udiv i32 %rounds.raised, %teams
%leads = icmp eq i32 %part, 0
%chunk.some = icmp ugt i32 %chunk.units, 0
%top.over = icmp ugt i32 %top, 64
%top.fails = and i1 %expert, %top.over
%down.straddle = urem i32 %hidden.nonzero, %unit
%down.straddles = icmp ne i32 %down.straddle, 0
%down.fails = and i1 %down, %down.straddles
%fails.some = or i1 %top.fails, %down.fails
%fails.none = xor i1 %fails.some, true
%takes = and i1 %chunk.some, %fails.none
; A plain sum with fewer groups of rows than workgroups, whose inputs fit one
; chunk, splits each row's units across workgroups too: each takes a group of
; rows and a slice of their units, leaves its parts in the split scratch, and
; after a grid barrier every thread adds up rows.
%split.fewer = icmp ult i32 %row.groups, %groups
%split.able = icmp ne ptr addrspace(1) %split.scratch, null
%split.whole = icmp eq i32 %chunk.units, %units
%split.plain = icmp eq i32 %mode, 0
%split.a = and i1 %split.fewer, %split.able
%split.b = and i1 %split.whole, %split.plain
%split = and i1 %split.a, %split.b
%slices = udiv i32 %groups, %row.groups
%split.row.group = urem i32 %group, %row.groups
%split.slice = udiv i32 %group, %row.groups
%split.live = icmp ult i32 %split.slice, %slices
%split.step.waves = mul i32 %slices, %waves
%split.slice.waves = mul i32 %split.slice, %waves
%split.first.unit = add i32 %split.slice.waves, %wave
%split.row.lane0 = mul i32 %split.row.group, %width
%split.row = add i32 %split.row.lane0, %lane
%split.row.in = icmp ult i32 %split.row, %rows
%split.row.live = and i1 %split.live, %split.row.in
%split.row.safe = select i1 %split.row.live, i32 %split.row, i32 0
%split.row.wide = zext i32 %split.row.safe to i64
%split.row.index = mul i64 %split.row.wide, %terms.wide
%split.row.base = add i64 %weight.base, %split.row.index
%split.threads.id.base = mul i32 %group, %block
%split.thread = add i32 %split.threads.id.base, %lid
%position.end = add i32 %out.begin, %out.span
br i1 %takes, label %position.loop, label %exit
position.loop:
%position.index = phi i32 [ %out.begin, %entry ], [ %position.next, %chunk.done ]
%position.more = icmp ult i32 %position.index, %position.end
br i1 %position.more, label %position.step, label %exit
position.step:
%position = zext i32 %position.index to i64
br i1 %split, label %split.stage, label %position.route
position.route:
br i1 %expert, label %route.entry, label %chunk.enter
split.stage:
%ss.c = phi i32 [ %lid, %position.step ], [ %ss.c.next, %split.stage.step ]
%ss.more = icmp ult i32 %ss.c, %terms
br i1 %ss.more, label %split.stage.step, label %split.staged
split.stage.step:
%ss.k.period = udiv i32 %ss.c, %in.share.nonzero
%ss.k.within = urem i32 %ss.c, %in.share.nonzero
%ss.k.base = mul i32 %ss.k.period, %in.period
%ss.k.placed = add i32 %ss.k.base, %ss.k.within
%ss.k.periodic = icmp ne i32 %in.period, 0
%ss.k.share = select i1 %ss.k.periodic, i32 %ss.k.placed, i32 %ss.c
%ss.k.global = add i32 %ss.k.share, %in.first
%ss.channel = zext i32 %ss.k.global to i64
%ss.index = mul i64 %ss.channel, %in.length.wide
%ss.at = add i64 %ss.index, %position
%ss.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %ss.at
%ss.value = load double, ptr addrspace(1) %ss.ptr, align 8
%ss.slot = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %ss.c
store double %ss.value, ptr addrspace(3) %ss.slot, align 8
%ss.c.next = add i32 %ss.c, %block
br label %split.stage
split.staged:
call void @recipe.local.barrier()
br label %split.unit
split.unit:
%su = phi i32 [ %split.first.unit, %split.staged ], [ %su.next, %split.unit.step ]
%su.sum = phi RECIPE_STATE [ %state.zero, %split.staged ], [ %su.sum.next, %split.unit.step ]
%su.more = icmp ult i32 %su, %units
%su.go = and i1 %su.more, %split.live
br i1 %su.go, label %split.unit.step, label %split.unit.done
split.unit.step:
%su.start = mul i32 %su, %unit
%su.start.wide = zext i32 %su.start to i64
%su.index = add i64 %split.row.base, %su.start.wide
%su.x = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %su.start
%su.run = call RECIPE_STATE @recipe.model.dot.run(ptr addrspace(1) %weights, i64 %su.index, i32 %node, ptr addrspace(3) %su.x, i32 1, i64 %terms.wide)
%su.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %su.sum, RECIPE_STATE %su.run)
%su.next = add i32 %su, %split.step.waves
br label %split.unit
split.unit.done:
%sp.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %parts, i32 %lid
store RECIPE_STATE %su.sum, ptr addrspace(3) %sp.ptr, align RECIPE_STATE_ALIGN
call void @recipe.local.barrier()
%sp.lead = icmp eq i32 %wave, 0
%sp.write = and i1 %sp.lead, %split.row.live
br i1 %sp.write, label %split.parts, label %split.written
split.parts:
%sw = phi i32 [ 1, %split.unit.done ], [ %sw.next, %split.parts.step ]
%sw.total = phi RECIPE_STATE [ %su.sum, %split.unit.done ], [ %sw.total.next, %split.parts.step ]
%sw.more = icmp ult i32 %sw, %waves
br i1 %sw.more, label %split.parts.step, label %split.parts.store
split.parts.step:
%sw.lane0 = mul i32 %sw, %width
%sw.slot = add i32 %sw.lane0, %lane
%sw.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %parts, i32 %sw.slot
%sw.value = load RECIPE_STATE, ptr addrspace(3) %sw.ptr, align RECIPE_STATE_ALIGN
%sw.total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sw.total, RECIPE_STATE %sw.value)
%sw.next = add i32 %sw, 1
br label %split.parts
split.parts.store:
%sg.slice = mul i32 %split.slice, %rows
%sg.at = add i32 %sg.slice, %split.row
%sg.ptr = getelementptr RECIPE_STATE, ptr addrspace(1) %split.scratch, i32 %sg.at
store RECIPE_STATE %sw.total, ptr addrspace(1) %sg.ptr, align RECIPE_STATE_ALIGN
br label %split.written
split.written:
call void @grid_barrier(i32 %threads)
br label %split.gather
split.gather:
%sr = phi i32 [ %split.thread, %split.written ], [ %sr.next, %split.gathered ]
%sr.more = icmp ult i32 %sr, %rows
br i1 %sr.more, label %split.gather.row, label %split.done
split.gather.row:
%sa = phi i32 [ 0, %split.gather ], [ %sa.next, %split.gather.add ]
%sa.total = phi RECIPE_STATE [ %state.zero, %split.gather ], [ %sa.total.next, %split.gather.add ]
%sa.more = icmp ult i32 %sa, %slices
br i1 %sa.more, label %split.gather.add, label %split.gathered.store
split.gather.add:
%sa.slice = mul i32 %sa, %rows
%sa.at = add i32 %sa.slice, %sr
%sa.ptr = getelementptr RECIPE_STATE, ptr addrspace(1) %split.scratch, i32 %sa.at
%sa.value = load volatile RECIPE_STATE, ptr addrspace(1) %sa.ptr, align RECIPE_STATE_ALIGN
%sa.total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sa.total, RECIPE_STATE %sa.value)
%sa.next = add i32 %sa, 1
br label %split.gather.row
split.gathered.store:
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 true, i1 true, i32 %sr, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position, i64 %weight.base, RECIPE_STATE %sa.total, i32 %row.first, i32 %row.period, i32 %row.share )
br label %split.gathered
split.gathered:
%sr.next = add i32 %sr, %threads
br label %split.gather
split.done:
call void @grid_barrier(i32 %threads)
br label %chunk.done
; The first wave lists the experts the position routed to in ascending order,
; as the row body does; a slot no expert fills reads expert 0 and adds nothing.
route.entry:
%route.share.over = add i32 %experts, %width
%route.share.raised = sub i32 %route.share.over, 1
%route.share = udiv i32 %route.share.raised, %width
%route.first = mul i32 %lane, %route.share
%route.first.end = add i32 %route.first, %route.share
%route.end.over = icmp ugt i32 %route.first.end, %experts
%route.end = select i1 %route.end.over, i32 %experts, i32 %route.first.end
br i1 %wave.first, label %route.clear, label %route.counted
route.clear:
%clear.c = phi i32 [ %lane, %route.entry ], [ %clear.c.next, %route.clear.step ]
%clear.more = icmp ult i32 %clear.c, %top
br i1 %clear.more, label %route.clear.step, label %route.count.entry
route.clear.step:
%clear.id = getelementptr i32, ptr addrspace(3) %ids, i32 %clear.c
store i32 0, ptr addrspace(3) %clear.id, align 4
%clear.scale = getelementptr RECIPE_STATE, ptr addrspace(3) %scales, i32 %clear.c
store RECIPE_STATE %state.zero, ptr addrspace(3) %clear.scale, align RECIPE_STATE_ALIGN
%clear.c.next = add i32 %clear.c, %width
br label %route.clear
route.count.entry:
br label %route.count
route.count:
%route.ce = phi i32 [ %route.first, %route.count.entry ], [ %route.ce.next, %route.count.step ]
%route.count.value = phi i32 [ 0, %route.count.entry ], [ %route.count.next, %route.count.step ]
%route.count.more = icmp ult i32 %route.ce, %route.end
br i1 %route.count.more, label %route.count.step, label %route.count.done
route.count.step:
%route.ce.wide = zext i32 %route.ce to i64
%route.ce.row = mul i64 %route.ce.wide, %out.length.wide
%route.ce.index = add i64 %route.ce.row, %position
%route.ce.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %route.ce.index
%route.ce.weight = load double, ptr addrspace(1) %route.ce.ptr, align 8
%route.ce.zero = call i1 @recipe.oeq(double %route.ce.weight, double 0.0)
%route.ce.taken = xor i1 %route.ce.zero, true
%route.ce.add = zext i1 %route.ce.taken to i32
%route.count.next = add i32 %route.count.value, %route.ce.add
%route.ce.next = add i32 %route.ce, 1
br label %route.count
route.count.done:
%route.count.slot = getelementptr i32, ptr addrspace(3) %parts, i32 %lane
store i32 %route.count.value, ptr addrspace(3) %route.count.slot, align 4
br label %route.counted
route.counted:
call void @recipe.local.barrier()
br i1 %wave.first, label %route.prefix, label %route.written
route.prefix:
%route.pl = phi i32 [ 0, %route.counted ], [ %route.pl.next, %route.prefix.step ]
%route.offset = phi i32 [ 0, %route.counted ], [ %route.offset.next, %route.prefix.step ]
%route.prefix.more = icmp ult i32 %route.pl, %lane
br i1 %route.prefix.more, label %route.prefix.step, label %route.loop
route.prefix.step:
%route.pl.slot = getelementptr i32, ptr addrspace(3) %parts, i32 %route.pl
%route.pl.count = load i32, ptr addrspace(3) %route.pl.slot, align 4
%route.offset.next = add i32 %route.offset, %route.pl.count
%route.pl.next = add i32 %route.pl, 1
br label %route.prefix
route.loop:
%route.e = phi i32 [ %route.first, %route.prefix ], [ %route.e.next, %route.advance ]
%route.slot = phi i32 [ %route.offset, %route.prefix ], [ %route.slot.next, %route.advance ]
%route.more.e = icmp ult i32 %route.e, %route.end
%route.more.slot = icmp ult i32 %route.slot, %top
%route.more = and i1 %route.more.e, %route.more.slot
br i1 %route.more, label %route.step, label %route.written
route.step:
%route.e.wide = zext i32 %route.e to i64
%route.row = mul i64 %route.e.wide, %out.length.wide
%route.index = add i64 %route.row, %position
%route.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %route.index
%route.weight = load double, ptr addrspace(1) %route.ptr, align 8
%route.zero = call i1 @recipe.oeq(double %route.weight, double 0.0)
br i1 %route.zero, label %route.advance, label %route.take
route.take:
%route.id.ptr = getelementptr i32, ptr addrspace(3) %ids, i32 %route.slot
store i32 %route.e, ptr addrspace(3) %route.id.ptr, align 4
%route.scale = call RECIPE_STATE @recipe.decode(double %route.weight)
%route.scale.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %scales, i32 %route.slot
store RECIPE_STATE %route.scale, ptr addrspace(3) %route.scale.ptr, align RECIPE_STATE_ALIGN
br label %route.advance
route.advance:
%route.took = xor i1 %route.zero, true
%route.step.slot = zext i1 %route.took to i32
%route.slot.next = add i32 %route.slot, %route.step.slot
%route.e.next = add i32 %route.e, 1
br label %route.loop
route.written:
call void @recipe.local.barrier()
br label %chunk.enter
chunk.enter:
br label %chunk.loop
chunk.loop:
%chunk.base = phi i32 [ 0, %chunk.enter ], [ %chunk.next, %round.loop.end ]
%chunk.remaining = sub i32 %terms, %chunk.base
%chunk.over = icmp ugt i32 %chunk.remaining, %chunk.span
%chunk.terms = select i1 %chunk.over, i32 %chunk.span, i32 %chunk.remaining
%chunk.end = add i32 %chunk.base, %chunk.terms
%chunk.first = icmp eq i32 %chunk.base, 0
%chunk.last = icmp eq i32 %chunk.end, %terms
%chunk.units.here = udiv i32 %chunk.terms, %unit
br label %stage.loop
stage.loop:
%stage.c = phi i32 [ %lid, %chunk.loop ], [ %stage.c.next, %stage.step ]
%stage.more = icmp ult i32 %stage.c, %chunk.terms
br i1 %stage.more, label %stage.step, label %stage.done
stage.step:
%stage.k = add i32 %stage.c, %chunk.base
%stage.k.period = udiv i32 %stage.k, %in.share.nonzero
%stage.k.within = urem i32 %stage.k, %in.share.nonzero
%stage.k.base = mul i32 %stage.k.period, %in.period
%stage.k.placed = add i32 %stage.k.base, %stage.k.within
%stage.k.periodic = icmp ne i32 %in.period, 0
%stage.k.share = select i1 %stage.k.periodic, i32 %stage.k.placed, i32 %stage.k
%stage.k.global = add i32 %stage.k.share, %in.first
%stage.channel = zext i32 %stage.k.global to i64
%stage.index = mul i64 %stage.channel, %in.length.wide
%stage.at = add i64 %stage.index, %position
%stage.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.at
%stage.value = load double, ptr addrspace(1) %stage.ptr, align 8
%stage.slot = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %stage.c
store double %stage.value, ptr addrspace(3) %stage.slot, align 8
%stage.c.next = add i32 %stage.c, %block
br label %stage.loop
stage.done:
call void @recipe.local.barrier()
br label %round.loop
round.loop:
%round = phi i32 [ 0, %stage.done ], [ %round.next, %round.done ]
%round.more = icmp ult i32 %round, %rounds
br i1 %round.more, label %round.step, label %round.loop.end
round.step:
%slot.base = mul i32 %round, %teams
%slot = add i32 %slot.base, %team
%slot.stride = mul i32 %slot, %groups
%row.group = add i32 %slot.stride, %group
%row.group.in = icmp ult i32 %row.group, %row.groups
%team.live = and i1 %team.in, %row.group.in
%row.lane0 = mul i32 %row.group, %width
%row = add i32 %row.lane0, %lane
%row.in = icmp ult i32 %row, %rows
%row.live = and i1 %team.live, %row.in
%row.safe = select i1 %row.live, i32 %row, i32 0
%row.wide = zext i32 %row.safe to i64
; A gate or up table reads row r % hidden of the expert in slot r / hidden.
%in.slot = udiv i32 %row.safe, %hidden.nonzero
%in.slot.safe = select i1 %expert, i32 %in.slot, i32 0
%in.id.ptr = getelementptr i32, ptr addrspace(3) %ids, i32 %in.slot.safe
%in.id = load i32, ptr addrspace(3) %in.id.ptr, align 4
%in.id.wide = zext i32 %in.id to i64
%in.f = urem i32 %row.safe, %hidden.nonzero
%in.f.wide = zext i32 %in.f to i64
%in.erow.base = mul i64 %in.id.wide, %hidden.wide
%in.erow = add i64 %in.erow.base, %in.f.wide
%weight.row = select i1 %expert, i64 %in.erow, i64 %row.wide
%weight.row.index = mul i64 %weight.row, %terms.wide
%row.base = add i64 %weight.base, %weight.row.index
br label %unit.loop
unit.loop:
%u = phi i32 [ %part, %round.step ], [ %u.next, %unit.step ]
%sum = phi RECIPE_STATE [ %state.zero, %round.step ], [ %sum.next, %unit.step ]
%u.more = icmp ult i32 %u, %chunk.units.here
%u.go = and i1 %u.more, %team.live
br i1 %u.go, label %unit.step, label %unit.done
unit.step:
%u.start = mul i32 %u, %unit
%k0 = add i32 %chunk.base, %u.start
%k0.wide = zext i32 %k0 to i64
; A down table reads, for the inputs of slot s, row r of slot s's expert, and
; scales the part by the slot's routing weight.
%down.slot = udiv i32 %k0, %hidden.nonzero
%down.slot.safe = select i1 %down, i32 %down.slot, i32 0
%down.id.ptr = getelementptr i32, ptr addrspace(3) %ids, i32 %down.slot.safe
%down.id = load i32, ptr addrspace(3) %down.id.ptr, align 4
%down.scale.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %scales, i32 %down.slot.safe
%down.scale = load RECIPE_STATE, ptr addrspace(3) %down.scale.ptr, align RECIPE_STATE_ALIGN
%down.id.wide = zext i32 %down.id to i64
%down.erow.base = mul i64 %down.id.wide, %rows.wide
%down.erow = add i64 %down.erow.base, %row.wide
%down.row.index = mul i64 %down.erow, %hidden.wide
%down.slot.wide = zext i32 %down.slot.safe to i64
%down.slot.start = mul i64 %down.slot.wide, %hidden.wide
%down.within = sub i64 %k0.wide, %down.slot.start
%down.index.local = add i64 %down.row.index, %down.within
%down.index = add i64 %weight.base, %down.index.local
%plain.index = add i64 %row.base, %k0.wide
%run.index = select i1 %down, i64 %down.index, i64 %plain.index
%x = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %u.start
%run.length = select i1 %down, i64 %hidden.wide, i64 %terms.wide
%run = call RECIPE_STATE @recipe.model.dot.run(ptr addrspace(1) %weights, i64 %run.index, i32 %node, ptr addrspace(3) %x, i32 1, i64 %run.length)
%run.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %run, RECIPE_STATE %down.scale)
%run.part = select i1 %down, RECIPE_STATE %run.scaled, RECIPE_STATE %run
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %run.part)
%u.next = add i32 %u, %team.size
br label %unit.loop
unit.done:
%part.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %parts, i32 %lid
store RECIPE_STATE %sum, ptr addrspace(3) %part.ptr, align RECIPE_STATE_ALIGN
call void @recipe.local.barrier()
%store.now = and i1 %leads, %row.live
br i1 %store.now, label %gather.loop, label %round.done
gather.loop:
%g = phi i32 [ 1, %unit.done ], [ %g.next, %gather.step ]
%total = phi RECIPE_STATE [ %sum, %unit.done ], [ %total.next, %gather.step ]
%g.more = icmp ult i32 %g, %team.size
br i1 %g.more, label %gather.step, label %gather.store
gather.step:
%g.wave = add i32 %wave, %g
%g.lane0 = mul i32 %g.wave, %width
%g.slot = add i32 %g.lane0, %lane
%g.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %parts, i32 %g.slot
%g.value = load RECIPE_STATE, ptr addrspace(3) %g.ptr, align RECIPE_STATE_ALIGN
%total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total, RECIPE_STATE %g.value)
%g.next = add i32 %g, 1
br label %gather.loop
gather.store:
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last, i32 %row, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position, i64 %weight.base, RECIPE_STATE %total, i32 %row.first, i32 %row.period, i32 %row.share )
br label %round.done
round.done:
call void @recipe.local.barrier()
%round.next = add i32 %round, 1
br label %round.loop
round.loop.end:
%chunk.next = add i32 %chunk.base, %chunk.terms
%chunk.more = icmp ult i32 %chunk.next, %terms
br i1 %chunk.more, label %chunk.loop, label %chunk.done
chunk.done:
%position.next = add i32 %position.index, 1
br label %position.loop
exit:
%handled = select i1 %takes, i32 %out.span, i32 0
ret i32 %handled
}
; Rows of a sum over stored weights at the positions %out.begin to %out.begin +
; %out.span, for any stored format with a run dot. Mode 0 reads row r of the
; weight; mode 1 (an expert's gate or up table) reads row r % hidden of the
; expert the position routed to slot r / hidden; mode 2 (an expert's down
; table) reads, for the inputs of slot s, row r of slot s's expert and scales
; that slot's part by its routing weight. A wave takes as many whole rows as
; fill its lanes with units (a block, or a run of smaller blocks): lane l works
; on unit l of the wave's rows, every lane runs the same place in its unit at
; a time, and each adds its row's part into the wave's row sums.
define internal void @packed_rows_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %routing, i32 %rows, i32 %terms, i32 %in.length, i32 %out.length, i32 %out.begin, i32 %out.span,
i1 %has.bias, i1 %relu, i32 %threads, i64 %weight.base, i32 %decode, i32 %node, i32 %mode, i32 %hidden, i32 %experts, i32 %top, ptr addrspace(1) %split.scratch, i32 %row.first, i32 %row.period, i32 %row.share, i32 %in.first, i32 %in.period, i32 %in.share ) RECIPE_CONTRACTION_BODY { entry:
%lid = call i32 @recipe.local.id.x()
%in.share.zero = icmp eq i32 %in.share, 0
%in.share.nonzero = select i1 %in.share.zero, i32 1, i32 %in.share
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%width = call i32 @recipe.wavefront.width()
%waves = udiv i32 %block, %width
%wave = udiv i32 %lid, %width
%lane = urem i32 %lid, %width
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%terms.wide = zext i32 %terms to i64
%in.length.wide = zext i32 %in.length to i64
%out.length.wide = zext i32 %out.length to i64
%hidden.wide = zext i32 %hidden to i64
%rows.wide = zext i32 %rows to i64
%hidden.zero = icmp eq i32 %hidden, 0
%hidden.nonzero = select i1 %hidden.zero, i32 1, i32 %hidden
%out.end = add i32 %out.begin, %out.span
%cases = add i32 1, 0
%unit = call i32 @recipe.model.dot.run.length(i32 %node)
%expert = icmp ne i32 %mode, 0
%down = icmp eq i32 %mode, 2
; The tile holds the staged column, then per wave 32 row sums, 32 slot experts
; and 32 slot routing weights.
%tile.bytes = call i32 @recipe.tile.bytes()
%scratch.values = mul i32 %waves, 96
%scratch.room = mul i32 %scratch.values, RECIPE_STATE_ALIGN
%tile.fits = icmp ugt i32 %tile.bytes, %scratch.room
%tile.left = sub i32 %tile.bytes, %scratch.room
%tile.room = select i1 %tile.fits, i32 %tile.left, i32 0
%tile.some = icmp ugt i32 %tile.room, 16
%tile.clear = sub i32 %tile.room, 16
%tile.usable = select i1 %tile.some, i32 %tile.clear, i32 0
%tile.values = udiv i32 %tile.usable, RECIPE_MODEL_BYTES
; Each unit of at least 32 values is staged with one value of padding.
%tile.padding.part = udiv i32 %tile.values, 32
%tile.padding = add i32 %tile.padding.part, 1
%tile.budget = sub i32 %tile.values, %tile.padding
%tile.blocks = udiv i32 %tile.budget, 256
%tile.chunk = mul i32 %tile.blocks, 256
%tile.chunk.some = icmp ugt i32 %tile.chunk, 256
%chunk.span = select i1 %tile.chunk.some, i32 %tile.chunk, i32 256
%x.pad = udiv i32 %chunk.span, 32
%x.values = add i32 %chunk.span, %x.pad
%x.bytes = mul i32 %x.values, RECIPE_MODEL_BYTES
%x.over = add i32 %x.bytes, 15
%x.aligned = and i32 %x.over, -16
%scratch = getelementptr i8, ptr addrspace(3) @contraction_tile, i32 %x.aligned
%wave.base = mul i32 %wave, 96
%sums.base = getelementptr RECIPE_STATE, ptr addrspace(3) %scratch, i32 %wave.base
%ids.base = getelementptr RECIPE_STATE, ptr addrspace(3) %sums.base, i32 32
%scales.base = getelementptr RECIPE_STATE, ptr addrspace(3) %sums.base, i32 64
%lane.sum = getelementptr RECIPE_STATE, ptr addrspace(3) %sums.base, i32 %lane
%lane.owns.m = icmp ult i32 %lane, 32
br label %position.loop
position.loop:
%position.index = phi i32 [ %out.begin, %entry ], [ %position.next, %position.done ]
%position.more = icmp ult i32 %position.index, %out.end
br i1 %position.more, label %position.step, label %exit
position.step:
%position = zext i32 %position.index to i64
br i1 %expert, label %route.clear, label %chunk.entry
; A slot no expert fills reads expert 0 and adds nothing, as the serial bodies do.
; Each lane scans an equal, contiguous share of the experts, so the experts a
; position routed to are listed in ascending order: a lane counts its routed
; experts, publishes the count, and writes its experts after the counts of the
; lanes before it.
route.clear:
%route.clear.in = icmp ult i32 %lane, %top
br i1 %route.clear.in, label %route.clear.store, label %route.count.entry
route.clear.store:
%route.c.id = getelementptr i32, ptr addrspace(3) %ids.base, i32 %lane
store i32 0, ptr addrspace(3) %route.c.id, align 4
%route.c.scale = getelementptr RECIPE_STATE, ptr addrspace(3) %scales.base, i32 %lane
store RECIPE_STATE %state.zero, ptr addrspace(3) %route.c.scale, align RECIPE_STATE_ALIGN
br label %route.count.entry
route.count.entry:
%route.share.over = add i32 %experts, %width
%route.share.raised = sub i32 %route.share.over, 1
%route.share = udiv i32 %route.share.raised, %width
%route.first = mul i32 %lane, %route.share
%route.first.end = add i32 %route.first, %route.share
%route.end.over = icmp ugt i32 %route.first.end, %experts
%route.end = select i1 %route.end.over, i32 %experts, i32 %route.first.end
br label %route.count
route.count:
%route.ce = phi i32 [ %route.first, %route.count.entry ], [ %route.ce.next, %route.count.step ]
%route.count.value = phi i32 [ 0, %route.count.entry ], [ %route.count.next, %route.count.step ]
%route.count.more = icmp ult i32 %route.ce, %route.end
br i1 %route.count.more, label %route.count.step, label %route.count.done
route.count.step:
%route.ce.wide = zext i32 %route.ce to i64
%route.ce.row = mul i64 %route.ce.wide, %out.length.wide
%route.ce.index = add i64 %route.ce.row, %position
%route.ce.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %route.ce.index
%route.ce.weight = load double, ptr addrspace(1) %route.ce.ptr, align 8
%route.ce.zero = call i1 @recipe.oeq(double %route.ce.weight, double 0.0)
%route.ce.taken = xor i1 %route.ce.zero, true
%route.ce.add = zext i1 %route.ce.taken to i32
%route.count.next = add i32 %route.count.value, %route.ce.add
%route.ce.next = add i32 %route.ce, 1
br label %route.count
route.count.done:
%route.count.slot = getelementptr i32, ptr addrspace(3) %sums.base, i32 %lane
store i32 %route.count.value, ptr addrspace(3) %route.count.slot, align 4
call void @recipe.local.barrier()
br label %route.prefix
route.prefix:
%route.pl = phi i32 [ 0, %route.count.done ], [ %route.pl.next, %route.prefix.step ]
%route.offset = phi i32 [ 0, %route.count.done ], [ %route.offset.next, %route.prefix.step ]
%route.prefix.more = icmp ult i32 %route.pl, %lane
br i1 %route.prefix.more, label %route.prefix.step, label %route.write.entry
route.prefix.step:
%route.pl.slot = getelementptr i32, ptr addrspace(3) %sums.base, i32 %route.pl
%route.pl.count = load i32, ptr addrspace(3) %route.pl.slot, align 4
%route.offset.next = add i32 %route.offset, %route.pl.count
%route.pl.next = add i32 %route.pl, 1
br label %route.prefix
route.write.entry:
br label %route.loop
route.loop:
%route.e = phi i32 [ %route.first, %route.write.entry ], [ %route.e.next, %route.advance ]
%route.slot = phi i32 [ %route.offset, %route.write.entry ], [ %route.slot.next, %route.advance ]
%route.more.e = icmp ult i32 %route.e, %route.end
%route.more.slot = icmp ult i32 %route.slot, %top
%route.more = and i1 %route.more.e, %route.more.slot
br i1 %route.more, label %route.step, label %route.written
route.step:
%route.e.wide = zext i32 %route.e to i64
%route.row = mul i64 %route.e.wide, %out.length.wide
%route.index = add i64 %route.row, %position
%route.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %route.index
%route.weight = load double, ptr addrspace(1) %route.ptr, align 8
%route.zero = call i1 @recipe.oeq(double %route.weight, double 0.0)
br i1 %route.zero, label %route.advance, label %route.take
route.take:
%route.id.ptr = getelementptr i32, ptr addrspace(3) %ids.base, i32 %route.slot
store i32 %route.e, ptr addrspace(3) %route.id.ptr, align 4
%route.scale = call RECIPE_STATE @recipe.decode(double %route.weight)
%route.scale.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %scales.base, i32 %route.slot
store RECIPE_STATE %route.scale, ptr addrspace(3) %route.scale.ptr, align RECIPE_STATE_ALIGN
br label %route.advance
route.advance:
%route.took = xor i1 %route.zero, true
%route.step.slot = zext i1 %route.took to i32
%route.slot.next = add i32 %route.slot, %route.step.slot
%route.e.next = add i32 %route.e, 1
br label %route.loop
route.written:
br label %chunk.entry
chunk.entry:
br label %chunk.loop
chunk.loop:
%chunk.base = phi i32 [ 0, %chunk.entry ], [ %chunk.next, %chunk.done ]
%chunk.remaining = sub i32 %terms, %chunk.base
%chunk.over = icmp ugt i32 %chunk.remaining, %chunk.span
%chunk.terms = select i1 %chunk.over, i32 %chunk.span, i32 %chunk.remaining
%chunk.end = add i32 %chunk.base, %chunk.terms
%chunk.first = icmp eq i32 %chunk.base, 0
%chunk.last = icmp eq i32 %chunk.end, %terms
%chunk.base.wide = zext i32 %chunk.base to i64
%upr = udiv i32 %chunk.terms, %unit
br label %stage.loop
; Value v of unit u sits at u * (unit + 1) + v: a lane reads its unit at
; constant offsets, and lanes on adjacent units land in adjacent banks.
stage.loop:
%stage.c = phi i32 [ %lid, %chunk.loop ], [ %stage.c.next, %stage.step ]
%stage.more = icmp ult i32 %stage.c, %chunk.terms
br i1 %stage.more, label %stage.step, label %stage.done
stage.step:
; A die summing a share of the inputs reads input k at its place: channels
; %in.first on, %in.share of every %in.period (or from %in.first on).
%stage.k = add i32 %stage.c, %chunk.base
%stage.k.period = udiv i32 %stage.k, %in.share.nonzero
%stage.k.within = urem i32 %stage.k, %in.share.nonzero
%stage.k.base = mul i32 %stage.k.period, %in.period
%stage.k.placed = add i32 %stage.k.base, %stage.k.within
%stage.k.periodic = icmp ne i32 %in.period, 0
%stage.k.share = select i1 %stage.k.periodic, i32 %stage.k.placed, i32 %stage.k
%stage.k.global = add i32 %stage.k.share, %in.first
%stage.c.wide = zext i32 %stage.k.global to i64
%stage.index = mul i64 %stage.c.wide, %in.length.wide
%stage.at = add i64 %stage.index, %position
%stage.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.at
%stage.value = load double, ptr addrspace(1) %stage.ptr, align 8
%stage.unit = udiv i32 %stage.c, %unit
%stage.within = urem i32 %stage.c, %unit
%stage.pitch = add i32 %unit, 1
%stage.row = mul i32 %stage.unit, %stage.pitch
%stage.place = add i32 %stage.row, %stage.within
%stage.slot = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %stage.place
store double %stage.value, ptr addrspace(3) %stage.slot, align 8
%stage.c.next = add i32 %stage.c, %block
br label %stage.loop
stage.done:
call void @recipe.local.barrier()
; Every chunk gives a wave the rows a whole chunk does, so one lane sums a
; row in every chunk with no barrier across workgroups between chunks.
%full.units = udiv i32 %chunk.span, %unit
%full.some = icmp ugt i32 %full.units, 0
%full.upr = select i1 %full.some, i32 %full.units, i32 1
%m.fill = udiv i32 %width, %full.upr
%m.some = icmp ugt i32 %m.fill, 0
%m = select i1 %m.some, i32 %m.fill, i32 1
%items = mul i32 %m, %upr
%sweeps.over = add i32 %items, %width
%sweeps.raised = sub i32 %sweeps.over, 1
%sweeps = udiv i32 %sweeps.raised, %width
%rowgroups.over = add i32 %rows, %m
%rowgroups.raised = sub i32 %rowgroups.over, 1
%rowgroups = udiv i32 %rowgroups.raised, %m
%jobs.over = add i32 %rowgroups, %waves
%jobs.raised = sub i32 %jobs.over, 1
%jobs = udiv i32 %jobs.raised, %waves
%lane.owns = icmp ult i32 %lane, %m
; Too few rows to give every wave work: each row splits into segments of up to
; a wave of units, each wave writes its segment's part, and after a grid
; barrier every row sums its parts in segment order.
%total.units = mul i32 %rows, %full.upr
%all.waves = udiv i32 %threads, %width
%target.jobs = mul i32 %all.waves, 2
%per.job.over = add i32 %total.units, %target.jobs
%per.job.raised = sub i32 %per.job.over, 1
%per.job.raw = udiv i32 %per.job.raised, %target.jobs
%per.job.some = icmp ugt i32 %per.job.raw, 0
%per.job.min = select i1 %per.job.some, i32 %per.job.raw, i32 1
%per.job.big = icmp ugt i32 %per.job.min, %width
%per.job = select i1 %per.job.big, i32 %width, i32 %per.job.min
%split.fewer = icmp ult i32 %per.job, %full.upr
%split.able = icmp ne ptr addrspace(1) %split.scratch, null
%split = and i1 %split.fewer, %split.able
br i1 %split, label %split.entry, label %job.loop
split.entry:
%segments.over = add i32 %upr, %per.job
%segments.raised = sub i32 %segments.over, 1
%segments = udiv i32 %segments.raised, %per.job
%split.jobs = mul i32 %rows, %segments
%split.wave.base = mul i32 %group, %waves
%split.wave = add i32 %split.wave.base, %wave
%reduce.start = udiv i32 %width, 2
br label %split.loop
split.loop:
%sj = phi i32 [ %split.wave, %split.entry ], [ %sj.next, %split.next ]
%sj.more = icmp ult i32 %sj, %split.jobs
br i1 %sj.more, label %split.step, label %split.gather
split.step:
%sj.row = udiv i32 %sj, %segments
%sj.segment = urem i32 %sj, %segments
%sj.first = mul i32 %sj.segment, %per.job
%sj.unit = add i32 %sj.first, %lane
%sj.unit.in = icmp ult i32 %sj.unit, %upr
%sj.lane.in = icmp ult i32 %lane, %per.job
%sj.active = and i1 %sj.unit.in, %sj.lane.in
%sj.part = call RECIPE_STATE @packed_unit_dot( ptr addrspace(1) %weights, i32 %node, i1 %expert, i1 %down, i32 %hidden.nonzero, i64 %hidden.wide, i64 %rows.wide, i64 %terms.wide, i64 %weight.base, i32 %chunk.base, i64 %chunk.base.wide, i32 %unit, i32 %cases, i32 %upr, ptr addrspace(3) %ids.base, ptr addrspace(3) %scales.base, i32 %sj.row, i32 %sj.unit, i1 %sj.active )
br label %split.reduce
split.reduce:
%sr.offset = phi i32 [ %reduce.start, %split.step ], [ %sr.offset.next, %split.reduce.step ]
%sr.value = phi RECIPE_STATE [ %sj.part, %split.step ], [ %sr.value.next, %split.reduce.step ]
%sr.more = icmp ugt i32 %sr.offset, 0
br i1 %sr.more, label %split.reduce.step, label %split.write
split.reduce.step:
%sr.partner.lane = xor i32 %lane, %sr.offset
%sr.partner.index = mul i32 %sr.partner.lane, 4
%sr.partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %sr.value, i32 %sr.partner.index)
%sr.value.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sr.value, RECIPE_STATE %sr.partner)
%sr.offset.next = udiv i32 %sr.offset, 2
br label %split.reduce
split.write:
%sj.leader = icmp eq i32 %lane, 0
br i1 %sj.leader, label %split.store, label %split.next
split.store:
%sj.slot = getelementptr RECIPE_STATE, ptr addrspace(1) %split.scratch, i32 %sj
store RECIPE_STATE %sr.value, ptr addrspace(1) %sj.slot, align RECIPE_STATE_ALIGN
br label %split.next
split.next:
%sj.next = add i32 %sj, %all.waves
br label %split.loop
split.gather:
call void @grid_barrier(i32 %threads)
%gather.base = mul i32 %group, %block
%gather.first = add i32 %gather.base, %lid
br label %gather.loop
gather.loop:
%gr = phi i32 [ %gather.first, %split.gather ], [ %gr.next, %gather.stored ]
%gr.more = icmp ult i32 %gr, %rows
br i1 %gr.more, label %gather.row, label %split.done
gather.row:
%gr.first = mul i32 %gr, %segments
br label %gather.sum
gather.sum:
%gs = phi i32 [ 0, %gather.row ], [ %gs.next, %gather.add ]
%gs.total = phi RECIPE_STATE [ %state.zero, %gather.row ], [ %gs.total.next, %gather.add ]
%gs.more = icmp ult i32 %gs, %segments
br i1 %gs.more, label %gather.add, label %gather.store
gather.add:
%gs.at = add i32 %gr.first, %gs
%gs.slot = getelementptr RECIPE_STATE, ptr addrspace(1) %split.scratch, i32 %gs.at
%gs.part = load RECIPE_STATE, ptr addrspace(1) %gs.slot, align RECIPE_STATE_ALIGN
%gs.total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %gs.total, RECIPE_STATE %gs.part)
%gs.next = add i32 %gs, 1
br label %gather.sum
gather.store:
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last, i32 %gr, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position, i64 %weight.base, RECIPE_STATE %gs.total, i32 %row.first, i32 %row.period, i32 %row.share )
br label %gather.stored
gather.stored:
%gr.next = add i32 %gr, %threads
br label %gather.loop
split.done:
; The next chunk or position writes the scratch again only after every row read it.
call void @grid_barrier(i32 %threads)
br label %chunk.done
job.loop:
%job = phi i32 [ %group, %stage.done ], [ %job.next, %job.done ]
%job.more = icmp ult i32 %job, %jobs
br i1 %job.more, label %job.step, label %chunk.done
job.step:
%rowgroup.base = mul i32 %job, %waves
%rowgroup = add i32 %rowgroup.base, %wave
%row0 = mul i32 %rowgroup, %m
br i1 %lane.owns, label %zero, label %sweep.loop
zero:
store RECIPE_STATE %state.zero, ptr addrspace(3) %lane.sum, align RECIPE_STATE_ALIGN
br label %sweep.loop
sweep.loop:
%sweep = phi i32 [ 0, %job.step ], [ 0, %zero ], [ %sweep.next, %sweep.done ]
%sweep.more = icmp ult i32 %sweep, %sweeps
br i1 %sweep.more, label %sweep.step, label %sweep.exit
sweep.step:
%sweep.base = mul i32 %sweep, %width
%item = add i32 %sweep.base, %lane
%row.local = udiv i32 %item, %upr
%unit.index = urem i32 %item, %upr
%row = add i32 %row0, %row.local
%item.in = icmp ult i32 %item, %items
%row.in = icmp ult i32 %row, %rows
%active = and i1 %item.in, %row.in
%part.add = call RECIPE_STATE @packed_unit_dot( ptr addrspace(1) %weights, i32 %node, i1 %expert, i1 %down, i32 %hidden.nonzero, i64 %hidden.wide, i64 %rows.wide, i64 %terms.wide, i64 %weight.base, i32 %chunk.base, i64 %chunk.base.wide, i32 %unit, i32 %cases, i32 %upr, ptr addrspace(3) %ids.base, ptr addrspace(3) %scales.base, i32 %row, i32 %unit.index, i1 %active )
br i1 %active, label %add, label %sweep.done
add:
%add.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %sums.base, i32 %row.local
%add.prior = atomicrmw fadd ptr addrspace(3) %add.ptr, RECIPE_STATE %part.add monotonic
br label %sweep.done
sweep.done:
%sweep.next = add i32 %sweep, 1
br label %sweep.loop
sweep.exit:
%out.row = add i32 %row0, %lane
%out.in = icmp ult i32 %out.row, %rows
%out.write = and i1 %lane.owns, %out.in
br i1 %out.write, label %write, label %job.done
write:
%sum = load RECIPE_STATE, ptr addrspace(3) %lane.sum, align RECIPE_STATE_ALIGN
call void @packed_row_store( ptr addrspace(1) %weights, ptr addrspace(1) %output, i32 %decode, i1 %has.bias, i1 %relu, i1 %chunk.first, i1 %chunk.last, i32 %out.row, i32 %rows, i32 %terms, i64 %out.length.wide, i64 %position, i64 %weight.base, RECIPE_STATE %sum, i32 %row.first, i32 %row.period, i32 %row.share )
br label %job.done
job.done:
%job.next = add i32 %job, %groups
br label %job.loop
chunk.done:
call void @recipe.local.barrier()
%chunk.next = add i32 %chunk.base, %chunk.terms
%chunk.more = icmp ult i32 %chunk.next, %terms
br i1 %chunk.more, label %chunk.loop, label %position.done
position.done:
%position.next = add i32 %position.index, 1
br label %position.loop
exit:
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
%sums = alloca [RECIPE_REGISTER_COUNT x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5) %state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %lid = call i32 @recipe.local.id.x() %group = call i32 @recipe.group.id.x() %block = call i32 @recipe.workgroup.size.x() %groups = udiv i32 %threads, %block
%in.channels.wide = zext i32 %in.channels to i64 %in.length.wide = zext i32 %in.length to i64 %out.channels.wide = zext i32 %out.channels to i64 %out.length.wide = zext i32 %out.length to i64 %rows.wide = zext i32 %rows to i64 %out.begin.wide = zext i32 %out.begin to i64 %out.span.wide = zext i32 %out.span to i64 %weight.base.wide = add i64 %weight.base, 0
%in.elements = mul i32 %in.channels, %in.length %in.elements.wide = mul i64 %in.channels.wide, %in.length.wide %out.elements.wide = mul i64 %out.channels.wide, %out.length.wide %is.conv = icmp ne i32 %kernel, 0 %span = select i1 %is.conv, i32 %kernel, i32 1 %terms = mul i32 %in.channels, %span %terms.wide = zext i32 %terms to i64 %m.total = mul i32 %rows, %out.span
%m.short = icmp ult i32 %tile.m, %m.total %m.tile.clamped = select i1 %m.short, i32 %tile.m, i32 %m.total %m.tile.empty = icmp eq i32 %m.tile.clamped, 0 %m.tile = select i1 %m.tile.empty, i32 1, i32 %m.tile.clamped %n.short = icmp ult i32 %tile.n, %out.channels %n.tile = select i1 %n.short, i32 %tile.n, i32 %out.channels %k.short = icmp ult i32 %tile.k, %terms %k.tile = select i1 %k.short, i32 %tile.k, i32 %terms
%m.adjusted = add i32 %m.total, %m.tile %m.numerator = sub i32 %m.adjusted, 1 %m.tiles = udiv i32 %m.numerator, %m.tile %n.adjusted = add i32 %out.channels, %n.tile %n.numerator = sub i32 %n.adjusted, 1 %n.tiles = udiv i32 %n.numerator, %n.tile %jobs = mul i32 %m.tiles, %n.tiles br label %job.loop job.loop:
%job = phi i32 [ %group, %entry ], [ %job.next, %job.done ] %job.more = icmp ult i32 %job, %jobs br i1 %job.more, label %job.step, label %exit job.step:
%m.group.short = icmp ult i32 %m.tiles, RECIPE_CONTRACTION_SWIZZLE_M %m.group.limit = select i1 %m.group.short, i32 %m.tiles, i32 RECIPE_CONTRACTION_SWIZZLE_M %group.width = mul i32 %m.group.limit, %n.tiles %group.index = udiv i32 %job, %group.width %m.group.base = mul i32 %group.index, %m.group.limit %m.group.remaining = sub i32 %m.tiles, %m.group.base %m.group.tail = icmp ult i32 %m.group.remaining, %m.group.limit %m.group.count = select i1 %m.group.tail, i32 %m.group.remaining, i32 %m.group.limit %group.local = urem i32 %job, %group.width %m.group.local = urem i32 %group.local, %m.group.count %m.tile.index = add i32 %m.group.base, %m.group.local %n.tile.index = udiv i32 %group.local, %m.group.count %m.base = mul i32 %m.tile.index, %m.tile %n.base = mul i32 %n.tile.index, %n.tile
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
%method.store = call i1 @contraction_store_lane(i1 %lane.store, i32 %lid)
%lane.n = udiv i32 %output.lane, %m.lanes %lane.m = urem i32 %output.lane, %m.lanes
%output.m.base = mul i32 %lane.m, RECIPE_REGISTER_M %output.n.base = mul i32 %lane.n, RECIPE_REGISTER_N br label %sum.init.loop sum.init.loop:
%sum.init = phi i32 [ 0, %job.step ], [ %sum.init.next, %sum.init.step ] %sum.init.more = icmp ult i32 %sum.init, RECIPE_REGISTER_COUNT br i1 %sum.init.more, label %sum.init.step, label %sum.init.done
sum.init.step: %sum.init.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %sum.init store RECIPE_STATE %state.zero, ptr addrspace(5) %sum.init.ptr, align RECIPE_STATE_ALIGN %sum.init.next = add i32 %sum.init, 1 br label %sum.init.loop
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
%a.tile.index = call i32 @contraction_a_index(i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br i1 %a.vector, label %load.a.vector, label %load.a.scalar
load.a.vector:
%a.vector.index = add i64 %a.row.base, %a.term.wide
%a.vector.source = getelementptr inbounds double, ptr addrspace(1) %input, i64 %a.vector.index
%a.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %a.vector.source, align 8
call void @contraction_stage_a_fragment(<RECIPE_FRAGMENT_K x double> %a.vector.value, i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br label %load.advance
load.a.scalar:
%a.loaded = call double @contraction_input( ptr addrspace(1) %input, i64 %a.row.base, i32 %a.position, i32 %a.term, i32 %span, i32 %in.length, i1 %is.conv )
br i1 %a.gate, label %load.a.activation, label %load.a.ready
load.a.activation:
%a.activation.channel = mul i64 %a.term.wide, %in.length.wide %a.activation.local = add i64 %a.activation.channel, %a.position.wide %a.activation.index = add i64 %a.row.base, %a.activation.local %a.activation.ptr = getelementptr inbounds double, ptr addrspace(1) %activation, i64 %a.activation.index %a.activation.value = load double, ptr addrspace(1) %a.activation.ptr, align 2 %a.activation.positive = call i1 @recipe.ogt(double %a.activation.value, double 0.0) %a.gated = select i1 %a.activation.positive, double %a.loaded, double 0.0
br label %load.a.ready
load.a.ready:
%a.value = phi double [ %a.loaded, %load.a.scalar ], [ %a.gated, %load.a.activation ]
br label %load.store
load.b.step: %b.local = sub i32 %load, %a.count %b.n = udiv i32 %b.local, %b.rows %b.row = urem i32 %b.local, %b.rows %b.k = mul i32 %b.row, %b.width %b.n.wide = zext i32 %b.n to i64 %b.k.wide = zext i32 %b.k to i64 %b.channel = add i64 %n.base.wide, %b.n.wide %b.channel.base = mul i64 %b.channel, %terms.wide %b.term = add i64 %term.base.wide, %b.k.wide
%b.direct.index = add i64 %b.channel.base, %b.term %b.out.channels.wide = zext i32 %out.channels to i64 %b.transpose.base = mul i64 %b.term, %b.out.channels.wide %b.transpose.index = add i64 %b.transpose.base, %b.channel %b.index = select i1 %transpose, i64 %b.transpose.index, i64 %b.direct.index %b.tile.base = mul i32 %tile.m, %tile.k %b.tile.local = call i32 @contraction_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k) %b.tile.index = add i32 %b.tile.base, %b.tile.local
br i1 %b.vector, label %load.b.vector, label %load.b.scalar
load.b.vector:
%b.vector.source = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %b.index
%b.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %b.vector.source, align 8
call void @contraction_stage_b_terms(<RECIPE_FRAGMENT_K x double> %b.vector.value, i32 %b.k, i32 %b.n, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %load.advance
load.b.scalar:
br i1 %weight.packed, label %load.b.packed, label %load.b.direct
load.b.direct:
%b.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %b.index
%b.loaded = load double, ptr addrspace(1) %b.ptr, align 8
br label %load.b.ready
load.b.packed:
%b.decode.index = add i64 %weight.base.wide, %b.index
%b.decoded = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %b.decode.index, i32 %decode)
br label %load.b.ready
load.b.ready:
%b.value = phi double [ %b.loaded, %load.b.direct ], [ %b.decoded, %load.b.packed ]
br label %load.store
load.store: %load.value = phi double [ %a.value, %load.a.ready ], [ %b.value, %load.b.ready ] %load.tile.index = phi i32 [ %a.tile.index, %load.a.ready ], [ %b.tile.index, %load.b.ready ] %load.tile.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %load.tile.index store double %load.value, ptr addrspace(3) %load.tile.ptr, align 8
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
%store.register = phi i32 [ 0, %accumulate.done ], [ %store.register.next, %store.next ] %store.more = icmp ult i32 %store.register, RECIPE_REGISTER_COUNT br i1 %store.more, label %store.check, label %job.done
store.check: %store.output.m.raw = call i32 @contraction_output_m(i32 %lid, i32 %store.register, i32 %m.lanes) %store.output.n.raw = call i32 @contraction_output_n(i32 %lid, i32 %store.register, i32 %m.lanes) %store.register.valid = call i1 @contraction_output_register_valid(i32 %store.register)
%store.output.m.valid = icmp ult i32 %store.output.m.raw, %m.count %store.output.n.valid = icmp ult i32 %store.output.n.raw, %n.count %store.output.valid = and i1 %store.output.m.valid, %store.output.n.valid %store.lane.active = and i1 %method.store, %store.output.valid %store.active = and i1 %store.lane.active, %store.register.valid br i1 %store.active, label %store, label %store.next
store: %store.channel = add i32 %n.base, %store.output.n.raw %store.m.global = add i32 %m.base, %store.output.m.raw %store.m.global.wide = zext i32 %store.m.global to i64 %store.span.wide = zext i32 %out.span to i64 %store.row.wide = udiv i64 %store.m.global.wide, %store.span.wide %store.position.local.wide = urem i64 %store.m.global.wide, %store.span.wide %store.begin.wide = zext i32 %out.begin to i64 %store.position.wide = add i64 %store.position.local.wide, %store.begin.wide %store.output.row.base = mul i64 %store.row.wide, %out.elements.wide
%store.channel.wide = zext i32 %store.channel to i64 %store.output.channel.base = mul i64 %store.channel.wide, %out.length.wide %store.output.local = add i64 %store.output.channel.base, %store.position.wide %store.output.index = add i64 %store.output.row.base, %store.output.local %store.output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %store.output.index
%store.bias.base = mul i64 %out.channels.wide, %terms.wide %store.channel.offset = zext i32 %store.channel to i64 %store.bias.index = add i64 %store.bias.base, %store.channel.offset
br i1 %weight.packed, label %store.bias.packed, label %store.bias.direct
store.bias.direct:
%store.bias.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %store.bias.index
%store.bias.loaded = load double, ptr addrspace(1) %store.bias.ptr, align 8
br label %store.bias.ready
store.bias.packed:
%store.bias.decode.index = add i64 %weight.base.wide, %store.bias.index
%store.bias.decoded = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %store.bias.decode.index, i32 %decode)
br label %store.bias.ready
store.bias.ready:
%store.bias = phi double [ %store.bias.loaded, %store.bias.direct ], [ %store.bias.decoded, %store.bias.packed ]
%store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %store.register %store.sum.wide = load RECIPE_STATE, ptr addrspace(5) %store.sum.ptr, align RECIPE_STATE_ALIGN %store.sum = call double @recipe.encode(RECIPE_STATE %store.sum.wide)
%store.biased = call double @recipe.add(double %store.sum, double %store.bias) %store.raw = select i1 %has.bias, double %store.biased, double %store.sum %store.forward = xor i1 %reverse, true %store.activate = and i1 %relu, %store.forward %store.positive = call i1 @recipe.ogt(double %store.raw, double 0.0) %store.activated = select i1 %store.positive, double %store.raw, double 0.0 %store.result = select i1 %store.activate, double %store.activated, double %store.raw %store.prior = load double, ptr addrspace(1) %store.output.ptr, align 2 %store.accumulated = call double @recipe.add(double %store.prior, double %store.result) %store.value = select i1 %accumulate, double %store.accumulated, double %store.result store double %store.value, ptr addrspace(1) %store.output.ptr, align 8 br label %store.next
store.next: %store.register.next = add i32 %store.register, 1 br label %store.loop job.done: %job.next = add i32 %job, %groups br label %job.loop exit: ret void }
define internal void @pool_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %p, i32 %from, i32 %to, i32 %size, i32 %channels, i1 %store.index ) #1 { entry: %from.wide = zext i32 %from to i64 %to.wide = zext i32 %to to i64 %size.wide = zext i32 %size to i64 %channels.wide = zext i32 %channels to i64
%length = udiv i64 %from.wide, %channels.wide %pooled.length = udiv i64 %to.wide, %channels.wide %row = udiv i64 %p, %to.wide %out = urem i64 %p, %to.wide
%channel = udiv i64 %out, %pooled.length %spatial = urem i64 %out, %pooled.length %start = mul i64 %spatial, %size.wide
%candidate.end = add i64 %start, %size.wide %short = icmp ult i64 %candidate.end, %length
%end = select i1 %short, i64 %candidate.end, i64 %length %row.base = mul i64 %row, %from.wide
%channel.local = mul i64 %channel, %length %input.base = add i64 %row.base, %channel.local br label %loop loop:
%i = phi i64 [ %start, %entry ], [ %next, %step ]
%maximum = phi double [ 0xFFF0000000000000, %entry ], [ %maximum.next, %step ]
%maximum.index = phi i64 [ %start, %entry ], [ %maximum.index.next, %step ] %more = icmp ult i64 %i, %end
br i1 %more, label %step, label %done step: %index = add i64 %input.base, %i
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %index
%value = load double, ptr addrspace(1) %input.ptr, align 8 %greater = call i1 @recipe.ogt(double %value, double %maximum)
%maximum.next = select i1 %greater, double %value, double %maximum
%maximum.index.next = select i1 %greater, i64 %index, i64 %maximum.index %next = add i64 %i, 1 br label %loop done:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p
store double %maximum, ptr addrspace(1) %output.ptr, align 8
br i1 %store.index, label %store.argmax, label %pool.exit
store.argmax:
%context.ptr = getelementptr inbounds i64, ptr addrspace(1) %context, i64 %p
store i64 %maximum.index, ptr addrspace(1) %context.ptr, align 8
br label %pool.exit
pool.exit:
ret void }
; Rotary embedding over the first %rotated channels of a fused QKV row: inside
; each head the channel pairs (i, i + dims/2) below %dims rotate by
; position * base^(-2i/dims). With %reverse the transpose rotation is added
; into %output, which makes the same body the adjoint pass.
define internal void @rope_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length,
	i32 %head.width, i32 %dims, i32 %rotated, double %base, double %yarn.mscale, double %yarn.factor, double %angle.chain, double %yarn.low, double %yarn.high, i1 %has.factors, i1 %reverse, i32 %position.origin ) #1 { entry: %channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %head.width.wide = zext i32 %head.width to i64 %dims.wide = zext i32 %dims to i64 %rotated.wide = zext i32 %rotated to i64 %per.row = mul i64 %channels.wide, %length.wide
%within = urem i64 %p, %per.row %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%local = urem i64 %channel, %head.width.wide %half = udiv i64 %dims.wide, 2
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %p
%value.model = load double, ptr addrspace(1) %input.ptr, align 8
%value = call RECIPE_STATE @recipe.decode(double %value.model)
%rotates = icmp ult i64 %channel, %rotated.wide %inside = icmp ult i64 %local, %dims.wide %active = and i1 %rotates, %inside
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%base.wide = call RECIPE_STATE @recipe.decode(double %base) %mscale.wide = call RECIPE_STATE @recipe.decode(double %yarn.mscale) %factor.wide = call RECIPE_STATE @recipe.decode(double %yarn.factor) %low.wide = call RECIPE_STATE @recipe.decode(double %yarn.low) %high.wide = call RECIPE_STATE @recipe.decode(double %yarn.high)
%unrotated.model = call double @recipe.encode(RECIPE_STATE %value)
br i1 %active, label %rotate, label %finish rotate: %upper = icmp uge i64 %local, %half
%local.upper = sub i64 %local, %half %index = select i1 %upper, i64 %local.upper, i64 %local
%half.stride = mul i64 %half, %length.wide %partner.up = add i64 %p, %half.stride %partner.down = sub i64 %p, %half.stride
%partner = select i1 %upper, i64 %partner.down, i64 %partner.up
%partner.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %partner
%other.model = load double, ptr addrspace(1) %partner.ptr, align 8
%other = call RECIPE_STATE @recipe.decode(double %other.model)
%two.index = mul i64 %index, 2 %two.index.i32 = trunc i64 %two.index to i32 %index.i32 = trunc i64 %index to i32 %two.index.value = call RECIPE_STATE @recipe.state.from.u32(i32 %two.index.i32) %index.value = call RECIPE_STATE @recipe.state.from.u32(i32 %index.i32)
br i1 %has.factors, label %factor.load, label %factor.one
factor.load:
%factor.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %index
%factor.model = load double, ptr addrspace(1) %factor.ptr, align 8
%factor.loaded = call RECIPE_STATE @recipe.decode(double %factor.model)
br label %factor.ready
factor.one:
br label %factor.ready
factor.ready:
%frequency.factor = phi RECIPE_STATE [ %factor.loaded, %factor.load ], [ %one, %factor.one ]
%dims.value = call RECIPE_STATE @recipe.state.from.u32(i32 %dims) %ratio = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %two.index.value, RECIPE_STATE %dims.value)
%log.base = call RECIPE_STATE @recipe.state.log(RECIPE_STATE %base.wide) %exponent.positive = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %ratio, RECIPE_STATE %log.base)
%exponent = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %exponent.positive) %frequency.raw = call RECIPE_STATE @recipe.state.exp(RECIPE_STATE %exponent)
%frequency.extrap = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %frequency.raw, RECIPE_STATE %frequency.factor)
%yarn.on = call i1 @recipe.state.ogt(RECIPE_STATE %factor.wide, RECIPE_STATE %one)
%ramp.span = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %high.wide, RECIPE_STATE %low.wide)
%ramp.offset = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %index.value, RECIPE_STATE %low.wide)
%ramp.raw = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %ramp.offset, RECIPE_STATE %ramp.span)
%ramp.low = call i1 @recipe.state.ogt(RECIPE_STATE %zero, RECIPE_STATE %ramp.raw)
%ramp.clamped.low = select i1 %ramp.low, RECIPE_STATE %zero, RECIPE_STATE %ramp.raw
%ramp.high = call i1 @recipe.state.ogt(RECIPE_STATE %ramp.clamped.low, RECIPE_STATE %one)
%ramp = select i1 %ramp.high, RECIPE_STATE %one, RECIPE_STATE %ramp.clamped.low
%interpolated = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %frequency.extrap, RECIPE_STATE %factor.wide)
%ramp.inverse = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %one, RECIPE_STATE %ramp)
%extrapolated.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %ramp.inverse, RECIPE_STATE %frequency.extrap)
%interpolated.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %ramp, RECIPE_STATE %interpolated)
%blended = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %extrapolated.part, RECIPE_STATE %interpolated.part)
%frequency = select i1 %yarn.on, RECIPE_STATE %blended, RECIPE_STATE %frequency.extrap
%position.local = trunc i64 %position to i32 %position.i32 = add i32 %position.local, %position.origin %position.value = call RECIPE_STATE @recipe.state.from.u32(i32 %position.i32) %angle = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %position.value, RECIPE_STATE %frequency)
%cos = call RECIPE_STATE @recipe.state.cos(RECIPE_STATE %angle) %sin = call RECIPE_STATE @recipe.state.sin(RECIPE_STATE %angle) %sin.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %sin)
%sin.signed = select i1 %reverse, RECIPE_STATE %sin.negative, RECIPE_STATE %sin %sin.signed.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %sin.signed)
%sin.term = select i1 %upper, RECIPE_STATE %sin.signed, RECIPE_STATE %sin.signed.negative
%cos.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %value, RECIPE_STATE %cos) %sin.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %other, RECIPE_STATE %sin.term)
%rotated.raw = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %cos.part, RECIPE_STATE %sin.part)
%rotated.direct = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %rotated.raw, RECIPE_STATE %mscale.wide)
; The chain angle, llama.cpp's rope: the pair's angle is the position times
; base^(-2/dims) taken one product at a time in the state, the yarn blend
; falls on the angle, the cosine and sine each carry the magnitude scale, and
; the rotation is two fused multiply-adds against the libm trig of the CPU.
; Under the chain the base argument already holds base^(-2/dims), taken on
; the host with the CPU's powf so every backend chains the same bits.
%chain.zero = call double @recipe.from.u32(i32 0)
%chain.on = call i1 @recipe.ogt(double %angle.chain, double %chain.zero)
br i1 %chain.on, label %chain, label %finish.direct
chain:
%chain.scale = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %base.wide, RECIPE_STATE %zero)
br label %chain.loop
chain.loop:
%chain.i = phi i64 [ 0, %chain ], [ %chain.i.next, %chain.step ]
%chain.theta = phi RECIPE_STATE [ %position.value, %chain ], [ %chain.theta.next, %chain.step ]
%chain.more = icmp ult i64 %chain.i, %index
br i1 %chain.more, label %chain.step, label %chain.done
chain.step:
%chain.theta.next = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.theta, RECIPE_STATE %chain.scale)
%chain.i.next = add i64 %chain.i, 1
br label %chain.loop
chain.done:
%chain.theta.extrap = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %chain.theta, RECIPE_STATE %frequency.factor)
%chain.freq.scale = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %one, RECIPE_STATE %factor.wide)
%chain.interp = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.freq.scale, RECIPE_STATE %chain.theta.extrap)
%chain.mix = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %one, RECIPE_STATE %ramp)
%chain.mix.inverse = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %one, RECIPE_STATE %chain.mix)
%chain.extrap.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.theta.extrap, RECIPE_STATE %chain.mix)
%chain.angle.yarn = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %chain.extrap.part, RECIPE_STATE %chain.interp, RECIPE_STATE %chain.mix.inverse)
%chain.angle = select i1 %yarn.on, RECIPE_STATE %chain.angle.yarn, RECIPE_STATE %chain.theta.extrap
%chain.cos.raw = call RECIPE_STATE @recipe.libm.cos(RECIPE_STATE %chain.angle)
%chain.sin.raw = call RECIPE_STATE @recipe.libm.sin(RECIPE_STATE %chain.angle)
%chain.cos = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.cos.raw, RECIPE_STATE %mscale.wide)
%chain.sin = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.sin.raw, RECIPE_STATE %mscale.wide)
%chain.sin.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %chain.sin)
%chain.sin.signed = select i1 %reverse, RECIPE_STATE %chain.sin.negative, RECIPE_STATE %chain.sin
%chain.sin.signed.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %chain.sin.signed)
%chain.sin.term = select i1 %upper, RECIPE_STATE %chain.sin.signed, RECIPE_STATE %chain.sin.signed.negative
%chain.other.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %other, RECIPE_STATE %chain.sin.term)
%chain.lower = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %chain.other.part, RECIPE_STATE %value, RECIPE_STATE %chain.cos)
%chain.upper.cos = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %value, RECIPE_STATE %chain.cos)
%chain.upper = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %chain.upper.cos, RECIPE_STATE %other, RECIPE_STATE %chain.sin.signed)
%rotated.chain = select i1 %upper, RECIPE_STATE %chain.upper, RECIPE_STATE %chain.lower
br label %finish.direct
finish.direct:
%rotated.value = phi RECIPE_STATE [ %rotated.direct, %factor.ready ], [ %rotated.chain, %chain.done ]
%rotated.model = call double @recipe.encode(RECIPE_STATE %rotated.value) br label %finish finish:
%result = phi double [ %unrotated.model, %entry ], [ %rotated.model, %finish.direct ]
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p
br i1 %reverse, label %accumulate, label %assign accumulate: %prior.model = load double, ptr addrspace(1) %output.ptr, align 8
%prior = call RECIPE_STATE @recipe.decode(double %prior.model) %result.wide = call RECIPE_STATE @recipe.decode(double %result) %sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %result.wide) %sum.model = call double @recipe.encode(RECIPE_STATE %sum) store double %sum.model, ptr addrspace(1) %output.ptr, align 8 ret void
assign: store double %result, ptr addrspace(1) %output.ptr, align 8 ret void }
; The adjoint twin: the same angles and coefficients, read from a state-typed delta
; and added into a state-typed previous, so nothing on the adjoint side is rounded.
define internal void @rope_body_adjoint( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length,
 i32 %head.width, i32 %dims, i32 %rotated, double %base, double %yarn.mscale, double %yarn.factor, double %angle.chain, double %yarn.low, double %yarn.high, i1 %has.factors, i1 %reverse ) #1 { entry: %channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %head.width.wide = zext i32 %head.width to i64 %dims.wide = zext i32 %dims to i64 %rotated.wide = zext i32 %rotated to i64 %per.row = mul i64 %channels.wide, %length.wide
%within = urem i64 %p, %per.row %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%local = urem i64 %channel, %head.width.wide %half = udiv i64 %dims.wide, 2
%input.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %input, i64 %p
%value = load RECIPE_STATE, ptr addrspace(1) %input.ptr, align RECIPE_STATE_ALIGN
%rotates = icmp ult i64 %channel, %rotated.wide %inside = icmp ult i64 %local, %dims.wide %active = and i1 %rotates, %inside
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%base.wide = call RECIPE_STATE @recipe.decode(double %base) %mscale.wide = call RECIPE_STATE @recipe.decode(double %yarn.mscale) %factor.wide = call RECIPE_STATE @recipe.decode(double %yarn.factor) %low.wide = call RECIPE_STATE @recipe.decode(double %yarn.low) %high.wide = call RECIPE_STATE @recipe.decode(double %yarn.high)
br i1 %active, label %rotate, label %finish rotate: %upper = icmp uge i64 %local, %half
%local.upper = sub i64 %local, %half %index = select i1 %upper, i64 %local.upper, i64 %local
%half.stride = mul i64 %half, %length.wide %partner.up = add i64 %p, %half.stride %partner.down = sub i64 %p, %half.stride
%partner = select i1 %upper, i64 %partner.down, i64 %partner.up
%partner.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %input, i64 %partner
%other = load RECIPE_STATE, ptr addrspace(1) %partner.ptr, align RECIPE_STATE_ALIGN
%two.index = mul i64 %index, 2 %two.index.i32 = trunc i64 %two.index to i32 %index.i32 = trunc i64 %index to i32 %two.index.value = call RECIPE_STATE @recipe.state.from.u32(i32 %two.index.i32) %index.value = call RECIPE_STATE @recipe.state.from.u32(i32 %index.i32)
br i1 %has.factors, label %factor.load, label %factor.one
factor.load:
%factor.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %index
%factor.model = load double, ptr addrspace(1) %factor.ptr, align 8
%factor.loaded = call RECIPE_STATE @recipe.decode(double %factor.model)
br label %factor.ready
factor.one:
br label %factor.ready
factor.ready:
%frequency.factor = phi RECIPE_STATE [ %factor.loaded, %factor.load ], [ %one, %factor.one ]
%dims.value = call RECIPE_STATE @recipe.state.from.u32(i32 %dims) %ratio = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %two.index.value, RECIPE_STATE %dims.value)
%log.base = call RECIPE_STATE @recipe.state.log(RECIPE_STATE %base.wide) %exponent.positive = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %ratio, RECIPE_STATE %log.base)
%exponent = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %exponent.positive) %frequency.raw = call RECIPE_STATE @recipe.state.exp(RECIPE_STATE %exponent)
%frequency.extrap = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %frequency.raw, RECIPE_STATE %frequency.factor)
%yarn.on = call i1 @recipe.state.ogt(RECIPE_STATE %factor.wide, RECIPE_STATE %one)
%ramp.span = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %high.wide, RECIPE_STATE %low.wide)
%ramp.offset = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %index.value, RECIPE_STATE %low.wide)
%ramp.raw = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %ramp.offset, RECIPE_STATE %ramp.span)
%ramp.low = call i1 @recipe.state.ogt(RECIPE_STATE %zero, RECIPE_STATE %ramp.raw)
%ramp.clamped.low = select i1 %ramp.low, RECIPE_STATE %zero, RECIPE_STATE %ramp.raw
%ramp.high = call i1 @recipe.state.ogt(RECIPE_STATE %ramp.clamped.low, RECIPE_STATE %one)
%ramp = select i1 %ramp.high, RECIPE_STATE %one, RECIPE_STATE %ramp.clamped.low
%interpolated = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %frequency.extrap, RECIPE_STATE %factor.wide)
%ramp.inverse = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %one, RECIPE_STATE %ramp)
%extrapolated.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %ramp.inverse, RECIPE_STATE %frequency.extrap)
%interpolated.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %ramp, RECIPE_STATE %interpolated)
%blended = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %extrapolated.part, RECIPE_STATE %interpolated.part)
%frequency = select i1 %yarn.on, RECIPE_STATE %blended, RECIPE_STATE %frequency.extrap
%position.i32 = trunc i64 %position to i32 %position.value = call RECIPE_STATE @recipe.state.from.u32(i32 %position.i32) %angle = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %position.value, RECIPE_STATE %frequency)
%cos = call RECIPE_STATE @recipe.state.cos(RECIPE_STATE %angle) %sin = call RECIPE_STATE @recipe.state.sin(RECIPE_STATE %angle) %sin.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %sin)
%sin.signed = select i1 %reverse, RECIPE_STATE %sin.negative, RECIPE_STATE %sin %sin.signed.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %sin.signed)
%sin.term = select i1 %upper, RECIPE_STATE %sin.signed, RECIPE_STATE %sin.signed.negative
%cos.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %value, RECIPE_STATE %cos) %sin.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %other, RECIPE_STATE %sin.term)
%rotated.raw = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %cos.part, RECIPE_STATE %sin.part)
%rotated.direct = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %rotated.raw, RECIPE_STATE %mscale.wide)
; The chain angle, llama.cpp's rope: the pair's angle is the position times
; base^(-2/dims) taken one product at a time in the state, the yarn blend
; falls on the angle, the cosine and sine each carry the magnitude scale, and
; the rotation is two fused multiply-adds against the libm trig of the CPU.
; Under the chain the base argument already holds base^(-2/dims), taken on
; the host with the CPU's powf so every backend chains the same bits.
%chain.zero = call double @recipe.from.u32(i32 0)
%chain.on = call i1 @recipe.ogt(double %angle.chain, double %chain.zero)
br i1 %chain.on, label %chain, label %finish.direct
chain:
%chain.scale = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %base.wide, RECIPE_STATE %zero)
br label %chain.loop
chain.loop:
%chain.i = phi i64 [ 0, %chain ], [ %chain.i.next, %chain.step ]
%chain.theta = phi RECIPE_STATE [ %position.value, %chain ], [ %chain.theta.next, %chain.step ]
%chain.more = icmp ult i64 %chain.i, %index
br i1 %chain.more, label %chain.step, label %chain.done
chain.step:
%chain.theta.next = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.theta, RECIPE_STATE %chain.scale)
%chain.i.next = add i64 %chain.i, 1
br label %chain.loop
chain.done:
%chain.theta.extrap = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %chain.theta, RECIPE_STATE %frequency.factor)
%chain.freq.scale = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %one, RECIPE_STATE %factor.wide)
%chain.interp = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.freq.scale, RECIPE_STATE %chain.theta.extrap)
%chain.mix = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %one, RECIPE_STATE %ramp)
%chain.mix.inverse = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %one, RECIPE_STATE %chain.mix)
%chain.extrap.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.theta.extrap, RECIPE_STATE %chain.mix)
%chain.angle.yarn = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %chain.extrap.part, RECIPE_STATE %chain.interp, RECIPE_STATE %chain.mix.inverse)
%chain.angle = select i1 %yarn.on, RECIPE_STATE %chain.angle.yarn, RECIPE_STATE %chain.theta.extrap
%chain.cos.raw = call RECIPE_STATE @recipe.libm.cos(RECIPE_STATE %chain.angle)
%chain.sin.raw = call RECIPE_STATE @recipe.libm.sin(RECIPE_STATE %chain.angle)
%chain.cos = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.cos.raw, RECIPE_STATE %mscale.wide)
%chain.sin = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %chain.sin.raw, RECIPE_STATE %mscale.wide)
%chain.sin.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %chain.sin)
%chain.sin.signed = select i1 %reverse, RECIPE_STATE %chain.sin.negative, RECIPE_STATE %chain.sin
%chain.sin.signed.negative = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %chain.sin.signed)
%chain.sin.term = select i1 %upper, RECIPE_STATE %chain.sin.signed, RECIPE_STATE %chain.sin.signed.negative
%chain.other.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %other, RECIPE_STATE %chain.sin.term)
%chain.lower = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %chain.other.part, RECIPE_STATE %value, RECIPE_STATE %chain.cos)
%chain.upper.cos = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %value, RECIPE_STATE %chain.cos)
%chain.upper = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %chain.upper.cos, RECIPE_STATE %other, RECIPE_STATE %chain.sin.signed)
%rotated.chain = select i1 %upper, RECIPE_STATE %chain.upper, RECIPE_STATE %chain.lower
br label %finish.direct
finish.direct:
%rotated.value = phi RECIPE_STATE [ %rotated.direct, %factor.ready ], [ %rotated.chain, %chain.done ]
br label %finish finish:
%result = phi RECIPE_STATE [ %value, %entry ], [ %rotated.value, %finish.direct ]
%output.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %output, i64 %p
br i1 %reverse, label %accumulate, label %assign accumulate: %prior = load RECIPE_STATE, ptr addrspace(1) %output.ptr, align RECIPE_STATE_ALIGN
%sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %result) store RECIPE_STATE %sum, ptr addrspace(1) %output.ptr, align RECIPE_STATE_ALIGN ret void
assign: store RECIPE_STATE %result, ptr addrspace(1) %output.ptr, align RECIPE_STATE_ALIGN ret void }
; Hyper-connection stream bodies. A stream row holds %lanes copies of
; %channels channels, lane l at channels [l * channels, (l + 1) * channels).
; Reverse bodies add into their adjoints; every element belongs to one thread.
; Output element %p of the widened batch copies the input channel of its lane.
define internal void @expand_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %lanes ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %per.row %within = urem i64 %p, %per.row
%lane.channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %channel = urem i64 %lane.channel, %channels.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %index.row = add i64 %row.base, %channel.base %index = add i64 %index.row, %position
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %index %value = load double, ptr addrspace(1) %input.ptr, align 8
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %value, ptr addrspace(1) %output.ptr, align 8 ret void }
; Source element %p of the narrow batch sums the adjoints of its lanes in lane order.
define internal void @expand_reverse_body( ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%row.base = mul i64 %row, %per.row %base = add i64 %row.base, %within br label %loop loop:
%lane = phi i64 [ 0, %entry ], [ %lane.next, %step ] %sum = phi RECIPE_STATE [ %state.zero, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %lane, %lanes.wide
br i1 %more, label %step, label %done step: %lane.offset = mul i64 %lane, %narrow %index = add i64 %base, %lane.offset
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %index %value = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %value) %lane.next = add i64 %lane, 1 br label %loop done:
%adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %p %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %sum) store RECIPE_STATE %total, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN ret void }
; Output element %p of the narrow batch is the gate-weighted mean of its lanes: the sum in lane order times 1 / lanes; without a gate every lane weighs one.
; A pick of lane l + 1 reads lane l alone; pick 0 is the gated mean of the lanes.
define internal void @read_forward_body( ptr addrspace(1) %stream, ptr addrspace(1) %gate, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated, i32 %pick ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow
%lanes.value = call double @recipe.from.u32(i32 %lanes) %scale = call double @recipe.div(double 1.0, double %lanes.value)
%row.base = mul i64 %row, %per.row %base = add i64 %row.base, %within
%picked = icmp ne i32 %pick, 0 br i1 %picked, label %take, label %base.ready
base.ready: br label %loop
take: %pick.lane = sub i32 %pick, 1 %pick.wide = zext i32 %pick.lane to i64 %pick.offset = mul i64 %pick.wide, %narrow %pick.index = add i64 %base, %pick.offset
%pick.ptr = getelementptr inbounds double, ptr addrspace(1) %stream, i64 %pick.index %pick.value = load double, ptr addrspace(1) %pick.ptr, align 8
%pick.output = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %pick.value, ptr addrspace(1) %pick.output, align 8 ret void
loop:
%lane = phi i64 [ 0, %base.ready ], [ %lane.next, %step ] %sum = phi double [ 0.0, %base.ready ], [ %sum.next, %step ] %more = icmp ult i64 %lane, %lanes.wide
br i1 %more, label %step, label %done step: %lane.offset = mul i64 %lane, %narrow %index = add i64 %base, %lane.offset
%stream.ptr = getelementptr inbounds double, ptr addrspace(1) %stream, i64 %index %value = load double, ptr addrspace(1) %stream.ptr, align 8
%gate.ptr = getelementptr inbounds double, ptr addrspace(1) %gate, i64 %index %gate.loaded = load double, ptr addrspace(1) %gate.ptr, align 8
%weight = select i1 %gated, double %gate.loaded, double 1.0 %product = call double @recipe.mul(double %weight, double %value)
%sum.next = call double @recipe.add(double %sum, double %product) %lane.next = add i64 %lane, 1 br label %loop done:
%mean = call double @recipe.mul(double %sum, double %scale)
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %mean, ptr addrspace(1) %output.ptr, align 8 ret void }
; Stream element %p receives gate * dh / lanes, and its gate receives stream * dh / lanes.
define internal void @read_reverse_body( ptr addrspace(1) %stream, ptr addrspace(1) %gate, ptr addrspace(1) %delta, ptr addrspace(1) %stream.adjoint, ptr addrspace(1) %gate.adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %per.row %within = urem i64 %p, %per.row
%lane.channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %channel = urem i64 %lane.channel, %channels.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %h.row = add i64 %row.base, %channel.base %h = add i64 %h.row, %position
%state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true) %lanes.value = call RECIPE_STATE @recipe.state.from.u32(i32 %lanes) %scale = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %state.one, RECIPE_STATE %lanes.value)
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %h %dh.loaded = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%dh = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dh.loaded, RECIPE_STATE %scale)
%gate.ptr = getelementptr inbounds double, ptr addrspace(1) %gate, i64 %p %gate.loaded = load double, ptr addrspace(1) %gate.ptr, align 8 %gate.wide = call RECIPE_STATE @recipe.decode(double %gate.loaded)
%weight = select i1 %gated, RECIPE_STATE %gate.wide, RECIPE_STATE %state.one %stream.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %weight, RECIPE_STATE %dh)
%stream.adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %stream.adjoint, i64 %p %stream.prior = load RECIPE_STATE, ptr addrspace(1) %stream.adjoint.ptr, align RECIPE_STATE_ALIGN
%stream.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %stream.prior, RECIPE_STATE %stream.term) store RECIPE_STATE %stream.sum, ptr addrspace(1) %stream.adjoint.ptr, align RECIPE_STATE_ALIGN
br i1 %gated, label %gate.pass, label %exit gate.pass:
%stream.ptr = getelementptr inbounds double, ptr addrspace(1) %stream, i64 %p %value = load double, ptr addrspace(1) %stream.ptr, align 8 %value.wide = call RECIPE_STATE @recipe.decode(double %value)
%gate.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %value.wide, RECIPE_STATE %dh)
%gate.adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gate.adjoint, i64 %p %gate.prior = load RECIPE_STATE, ptr addrspace(1) %gate.adjoint.ptr, align RECIPE_STATE_ALIGN
%gate.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %gate.prior, RECIPE_STATE %gate.term) store RECIPE_STATE %gate.sum, ptr addrspace(1) %gate.adjoint.ptr, align RECIPE_STATE_ALIGN br label %exit exit: ret void }
; Output element %p of the widened batch is its lane's write gate times the branch output channel.
; A pick of lane l + 1 writes the branch into lane l and zeros every other lane.
define internal void @outer_forward_body( ptr addrspace(1) %branch, ptr addrspace(1) %gate, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated, i32 %pick ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %per.row %within = urem i64 %p, %per.row
%lane.channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %channel = urem i64 %lane.channel, %channels.wide %lane = udiv i64 %lane.channel, %channels.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %y.row = add i64 %row.base, %channel.base %y = add i64 %y.row, %position
%gate.row = mul i64 %row, %lanes.wide %gate.lane = add i64 %gate.row, %lane %gate.lane.base = mul i64 %gate.lane, %length.wide %g = add i64 %gate.lane.base, %position
%branch.ptr = getelementptr inbounds double, ptr addrspace(1) %branch, i64 %y %value = load double, ptr addrspace(1) %branch.ptr, align 8
%gate.ptr = getelementptr inbounds double, ptr addrspace(1) %gate, i64 %g %gate.loaded = load double, ptr addrspace(1) %gate.ptr, align 8
%weight.gated = select i1 %gated, double %gate.loaded, double 1.0 %pick.lane = sub i32 %pick, 1 %pick.wide = zext i32 %pick.lane to i64 %pick.here = icmp eq i64 %lane, %pick.wide
%pick.weight = select i1 %pick.here, double 1.0, double 0.0 %pick.zero = call double @recipe.from.u1(i1 false) %pick.value = select i1 %pick.here, double %value, double %pick.zero %picked = icmp ne i32 %pick, 0 %weight = select i1 %picked, double %pick.weight, double %weight.gated
%product.raw = call double @recipe.mul(double %weight, double %value) %product = select i1 %picked, double %pick.value, double %product.raw
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %product, ptr addrspace(1) %output.ptr, align 8 ret void }
; Branch element %p sums gate * adjoint over its lanes in lane order.
define internal void @outer_reverse_branch_body( ptr addrspace(1) %gate, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes, i1 %gated ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %position = urem i64 %within, %length.wide
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%row.base = mul i64 %row, %per.row %base = add i64 %row.base, %within %gate.row = mul i64 %row, %lanes.wide br label %loop loop:
%lane = phi i64 [ 0, %entry ], [ %lane.next, %step ] %sum = phi RECIPE_STATE [ %state.zero, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %lane, %lanes.wide
br i1 %more, label %step, label %done step: %lane.offset = mul i64 %lane, %narrow %index = add i64 %base, %lane.offset
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %index %value = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%gate.lane = add i64 %gate.row, %lane %gate.lane.base = mul i64 %gate.lane, %length.wide %g = add i64 %gate.lane.base, %position
%gate.ptr = getelementptr inbounds double, ptr addrspace(1) %gate, i64 %g %gate.loaded = load double, ptr addrspace(1) %gate.ptr, align 8 %gate.wide = call RECIPE_STATE @recipe.decode(double %gate.loaded)
%weight = select i1 %gated, RECIPE_STATE %gate.wide, RECIPE_STATE %state.one %product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %weight, RECIPE_STATE %value)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %product) %lane.next = add i64 %lane, 1 br label %loop done:
%adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %p %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %sum) store RECIPE_STATE %total, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN ret void }
; Gate element %p (one per row, lane, and position) sums branch * adjoint over its channels in channel order.
define internal void @outer_reverse_gate_body( ptr addrspace(1) %branch, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %lanes ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %lanes.wide = zext i32 %lanes to i64
%narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %narrow, %lanes.wide %gates.row = mul i64 %lanes.wide, %length.wide %row = udiv i64 %p, %gates.row %within = urem i64 %p, %gates.row
%lane = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide %row.base = mul i64 %row, %per.row %lane.base = mul i64 %lane, %narrow
%delta.base.row = add i64 %row.base, %lane.base %delta.base = add i64 %delta.base.row, %position %branch.row = mul i64 %row, %narrow %branch.base = add i64 %branch.row, %position
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br label %loop loop: %channel = phi i64 [ 0, %entry ], [ %channel.next, %step ] %sum = phi RECIPE_STATE [ %state.zero, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %channel, %channels.wide
br i1 %more, label %step, label %done step: %channel.offset = mul i64 %channel, %length.wide %delta.index = add i64 %delta.base, %channel.offset %branch.index = add i64 %branch.base, %channel.offset
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %delta.index %value = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%branch.ptr = getelementptr inbounds double, ptr addrspace(1) %branch, i64 %branch.index %y = load double, ptr addrspace(1) %branch.ptr, align 8 %y.wide = call RECIPE_STATE @recipe.decode(double %y)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %y.wide, RECIPE_STATE %value) %sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %product) %channel.next = add i64 %channel, 1 br label %loop done:
%adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %p %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %sum) store RECIPE_STATE %total, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN ret void }
; Channel-group sum. Output element %p of a [rows][groups][length] batch sums the
; %width channels of its group at its position, in channel order.
define internal void @fold_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %output, i64 %p, i32 %groups, i32 %width, i32 %length ) #1 { entry:
%groups.wide = zext i32 %groups to i64 %width.wide = zext i32 %width to i64 %length.wide = zext i32 %length to i64
%narrow = mul i64 %groups.wide, %length.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %group = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%channels = mul i64 %groups.wide, %width.wide %wide = mul i64 %channels, %length.wide %row.base = mul i64 %row, %wide %first = mul i64 %group, %width.wide %first.base = mul i64 %first, %length.wide
%base.row = add i64 %row.base, %first.base %base = add i64 %base.row, %position br label %loop loop:
%c = phi i64 [ 0, %entry ], [ %c.next, %step ] %sum = phi double [ 0.0, %entry ], [ %sum.next, %step ] %more = icmp ult i64 %c, %width.wide br i1 %more, label %step, label %done
step: %offset = mul i64 %c, %length.wide %index = add i64 %base, %offset %input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %index %value = load double, ptr addrspace(1) %input.ptr, align 8
%sum.next = call double @recipe.add(double %sum, double %value) %c.next = add i64 %c, 1 br label %loop
done: %output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %sum, ptr addrspace(1) %output.ptr, align 8 ret void }
; Input element %p receives the adjoint of its group at its position.
define internal void @fold_reverse_body( ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %groups, i32 %width, i32 %length ) #1 { entry:
%groups.wide = zext i32 %groups to i64 %width.wide = zext i32 %width to i64 %length.wide = zext i32 %length to i64
%channels = mul i64 %groups.wide, %width.wide %wide = mul i64 %channels, %length.wide %row = udiv i64 %p, %wide %within = urem i64 %p, %wide %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%group = udiv i64 %channel, %width.wide %narrow = mul i64 %groups.wide, %length.wide %row.base = mul i64 %row, %narrow %group.base = mul i64 %group, %length.wide %index.row = add i64 %row.base, %group.base %index = add i64 %index.row, %position
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %index %value = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %p %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %value) store RECIPE_STATE %total, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN ret void }
; Causal depthwise convolution. Output element %p of a [rows][channels][length]
; batch sums %kernel positions of its own channel, tap j reading position
; t - (kernel - 1 - j) * dilation, with positions before the start reading zero.
define internal void @dconv_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %kernel, i32 %dilation, i32 %decode ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %kernel.wide = zext i32 %kernel to i64 %dilation.wide = zext i32 %dilation to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %base = add i64 %row.base, %channel.base %tap.base = mul i64 %channel, %kernel.wide
%reach = sub i64 %kernel.wide, 1 br label %loop loop: %tap = phi i64 [ 0, %entry ], [ %tap.next, %step ] %sum = phi double [ 0.0, %entry ], [ %sum.next, %step ]
%more = icmp ult i64 %tap, %kernel.wide br i1 %more, label %step, label %done step: %back = sub i64 %reach, %tap %back.scaled = mul i64 %back, %dilation.wide %source = sub i64 %position, %back.scaled
%valid = icmp sge i64 %source, 0 %source.clamped = select i1 %valid, i64 %source, i64 0 %index = add i64 %base, %source.clamped
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %index %loaded = load double, ptr addrspace(1) %input.ptr, align 8 %value = select i1 %valid, double %loaded, double 0.0
%tap.index = add i64 %tap.base, %tap %weight = call double @recipe.model.weight(ptr addrspace(1) %weights, i64 %tap.index, i32 %decode)
%product = call double @recipe.mul(double %weight, double %value) %sum.next = call double @recipe.add(double %sum, double %product) %tap.next = add i64 %tap, 1 br label %loop
done: %output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %sum, ptr addrspace(1) %output.ptr, align 8 ret void }
; One position of a depthwise causal convolution over a buffer that holds only
; its window: a tap behind the window reads the channel's history, the last
; (kernel - 1) * dilation inputs before the window, or zero when %fresh.
define internal void @dconv_window_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %history, i64 %p, i32 %channels, i32 %length, i32 %kernel, i32 %dilation, i32 %decode, i1 %fresh ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %kernel.wide = zext i32 %kernel to i64 %dilation.wide = zext i32 %dilation to i64
%channel = udiv i64 %p, %length.wide %position = urem i64 %p, %length.wide
%base = mul i64 %channel, %length.wide %tap.base = mul i64 %channel, %kernel.wide
%reach = sub i64 %kernel.wide, 1 %tail = mul i64 %reach, %dilation.wide %past.base = mul i64 %channel, %tail
br label %loop
loop: %tap = phi i64 [ 0, %entry ], [ %tap.next, %step ] %sum = phi double [ 0.0, %entry ], [ %sum.next, %step ]
%more = icmp ult i64 %tap, %kernel.wide br i1 %more, label %step, label %done
step: %back = sub i64 %reach, %tap %back.scaled = mul i64 %back, %dilation.wide %source = sub i64 %position, %back.scaled
%inside = icmp sge i64 %source, 0 %source.clamped = select i1 %inside, i64 %source, i64 0 %index = add i64 %base, %source.clamped
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %index %loaded = load double, ptr addrspace(1) %input.ptr, align 8
%past = add i64 %tail, %source %past.clamped = select i1 %inside, i64 0, i64 %past %past.index = add i64 %past.base, %past.clamped
%past.ptr = getelementptr inbounds double, ptr addrspace(1) %history, i64 %past.index %past.loaded = load double, ptr addrspace(1) %past.ptr, align 8
%past.value = select i1 %fresh, double 0.0, double %past.loaded %value = select i1 %inside, double %loaded, double %past.value
%tap.index = add i64 %tap.base, %tap %weight = call double @recipe.model.weight(ptr addrspace(1) %weights, i64 %tap.index, i32 %decode)
%product = call double @recipe.mul(double %weight, double %value) %sum.next = call double @recipe.add(double %sum, double %product) %tap.next = add i64 %tap, 1 br label %loop
done: %output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %sum, ptr addrspace(1) %output.ptr, align 8 ret void }
; After a window, channel %channel's history takes the last %tail inputs: from the
; window as far back as it reaches, then from the older history (zero when %fresh).
; One thread walks a channel in order, so each slot reads an older slot first.
define internal void @dconv_history_body( ptr addrspace(1) %input, ptr addrspace(1) %history, i64 %channel, i32 %length, i32 %span, i32 %tail, i1 %fresh ) #1 { entry:
%length.wide = zext i32 %length to i64 %tail.wide = zext i32 %tail to i64 %span.wide = zext i32 %span to i64
%base = mul i64 %channel, %length.wide %past.base = mul i64 %channel, %tail.wide %start = sub i64 %span.wide, %tail.wide
br label %loop
loop: %h = phi i64 [ 0, %entry ], [ %h.next, %step ] %more = icmp ult i64 %h, %tail.wide br i1 %more, label %step, label %done
step: %source = add i64 %start, %h %inside = icmp sge i64 %source, 0 %source.clamped = select i1 %inside, i64 %source, i64 0
%index = add i64 %base, %source.clamped %input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %index %loaded = load double, ptr addrspace(1) %input.ptr, align 8
%older = add i64 %h, %span.wide %older.inside = icmp ult i64 %older, %tail.wide %older.clamped = select i1 %older.inside, i64 %older, i64 0
%older.index = add i64 %past.base, %older.clamped %older.ptr = getelementptr inbounds double, ptr addrspace(1) %history, i64 %older.index %older.loaded = load double, ptr addrspace(1) %older.ptr, align 8
%older.kept = select i1 %fresh, double 0.0, double %older.loaded %value = select i1 %inside, double %loaded, double %older.kept
%slot = add i64 %past.base, %h %slot.ptr = getelementptr inbounds double, ptr addrspace(1) %history, i64 %slot store double %value, ptr addrspace(1) %slot.ptr, align 8
%h.next = add i64 %h, 1 br label %loop
done: ret void }
; Input element %p receives tap j times the adjoint at position t + (kernel - 1 - j) * dilation while that position exists.
define internal void @dconv_reverse_input_body( ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %kernel, i32 %dilation ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %kernel.wide = zext i32 %kernel to i64 %dilation.wide = zext i32 %dilation to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %within = urem i64 %p, %narrow %channel = udiv i64 %within, %length.wide %position = urem i64 %within, %length.wide
%row.base = mul i64 %row, %narrow %channel.base = mul i64 %channel, %length.wide %base = add i64 %row.base, %channel.base %tap.base = mul i64 %channel, %kernel.wide
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%reach = sub i64 %kernel.wide, 1 br label %loop loop: %tap = phi i64 [ 0, %entry ], [ %tap.next, %step ] %sum = phi RECIPE_STATE [ %state.zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i64 %tap, %kernel.wide br i1 %more, label %step, label %done step: %ahead.taps = sub i64 %reach, %tap %ahead = mul i64 %ahead.taps, %dilation.wide %target = add i64 %position, %ahead
%valid = icmp ult i64 %target, %length.wide %target.clamped = select i1 %valid, i64 %target, i64 0 %index = add i64 %base, %target.clamped
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %index %loaded = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN %value = select i1 %valid, RECIPE_STATE %loaded, RECIPE_STATE %state.zero
%tap.index = add i64 %tap.base, %tap %weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %tap.index %weight = load double, ptr addrspace(1) %weight.ptr, align 8 %weight.wide = call RECIPE_STATE @recipe.decode(double %weight)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %weight.wide, RECIPE_STATE %value) %sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %product) %tap.next = add i64 %tap, 1 br label %loop
done: %adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %p %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %sum) store RECIPE_STATE %total, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN ret void }
; Tap %p (channel * kernel + j) sums input * adjoint over every row and position in order and writes its gradient.
define internal void @dconv_reverse_weight_body( ptr addrspace(1) %input, ptr addrspace(1) %delta, ptr addrspace(1) %gradient, i64 %p, i32 %rows, i32 %channels, i32 %length, i32 %kernel, i32 %dilation, i32 %offset ) #1 { entry:
%kernel.wide = zext i32 %kernel to i64 %channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %dilation.wide = zext i32 %dilation to i64 %rows.wide = zext i32 %rows to i64 %offset.wide = zext i32 %offset to i64
%channel = udiv i64 %p, %kernel.wide %tap = urem i64 %p, %kernel.wide %narrow = mul i64 %channels.wide, %length.wide %channel.base = mul i64 %channel, %length.wide %reach = sub i64 %kernel.wide, 1 %shift.taps = sub i64 %reach, %tap %shift = mul i64 %shift.taps, %dilation.wide
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%count = mul i64 %rows.wide, %length.wide br label %loop loop: %step.index = phi i64 [ 0, %entry ], [ %step.next, %step ] %sum = phi RECIPE_STATE [ %state.zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i64 %step.index, %count br i1 %more, label %step, label %done step: %row = udiv i64 %step.index, %length.wide %position = urem i64 %step.index, %length.wide
%valid = icmp uge i64 %position, %shift %source = sub i64 %position, %shift %source.clamped = select i1 %valid, i64 %source, i64 0
%row.base = mul i64 %row, %narrow %base = add i64 %row.base, %channel.base %input.index = add i64 %base, %source.clamped %delta.index = add i64 %base, %position
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %input.index %loaded = load double, ptr addrspace(1) %input.ptr, align 8 %loaded.wide = call RECIPE_STATE @recipe.decode(double %loaded) %value = select i1 %valid, RECIPE_STATE %loaded.wide, RECIPE_STATE %state.zero
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %delta.index %incoming = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %value, RECIPE_STATE %incoming) %sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %product) %step.next = add i64 %step.index, 1 br label %loop
done: %gradient.index = add i64 %offset.wide, %p %gradient.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gradient, i64 %gradient.index store RECIPE_STATE %sum, ptr addrspace(1) %gradient.ptr, align RECIPE_STATE_ALIGN ret void }
; log(1 + exp(x)) taken on the negative side so a large x cannot overflow.
define internal double @softplus(double %x) #1 { entry: %magnitude = call double @recipe.abs(double %x)
%negative = call double @recipe.neg(double %magnitude) %exponential = call double @recipe.exp(double %negative)
%shifted = call double @recipe.add(double 1.0, double %exponential) %tail = call double @recipe.log(double %shifted)
%softplus.zero = call double @recipe.from.u1(i1 false) %positive = call i1 @recipe.ogt(double %x, double %softplus.zero) %linear = select i1 %positive, double %x, double %softplus.zero
%value = call double @recipe.add(double %linear, double %tail) ret double %value }
; One position of the gated delta rule for one head. The state at %work.base is
; read and written in place: S <- decay * S + write * k' (v - k S), and the
; output o = q S is stored when %store is set. Every sum walks the head in
; ascending order, so the position update never depends on the chunk it sits in.
; One value column of one position of the gated delta rule: the column's
; cells of the state at %work.base are read and written in place, and the
; column's output is stored when %store is set. Columns never read each other.
define internal void @delta_column( ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %k.base, i64 %v.base, i64 %q.base, i64 %o.base, i64 %work.base,
i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, i32 %column, double %decay, double %write, i1 %store, double %scale ) #1 { entry:
%time.wide = zext i32 %time to i64 %kwidth.wide = zext i32 %kwidth to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64 %column.wide = zext i32 %column to i64
br label %read.loop
read.loop: %read.i = phi i32 [ 0, %entry ], [ %read.next, %read.step ]
%read.sum = phi double [ 0.0, %entry ], [ %read.sum.next, %read.step ]
%read.more = icmp ult i32 %read.i, %kwidth br i1 %read.more, label %read.step, label %read.done
read.step: %read.i.wide = zext i32 %read.i to i64 %read.row = mul i64 %read.i.wide, %vwidth.wide %read.cell = add i64 %read.row, %column.wide %read.index = add i64 %work.base, %read.cell
%read.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %read.index
%read.state = load double, ptr addrspace(1) %read.pointer, align 8
%read.offset.row = mul i64 %read.i.wide, %length.wide %read.offset = add i64 %read.offset.row, %time.wide %read.key.index = add i64 %k.base, %read.offset
%read.key.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %read.key.index
%read.key = load double, ptr addrspace(1) %read.key.pointer, align 8
%read.product = call double @recipe.mul(double %read.key, double %read.state)
%read.sum.next = call double @recipe.add(double %read.sum, double %read.product) %read.next = add nuw i32 %read.i, 1 br label %read.loop
read.done: %value.row = mul i64 %column.wide, %length.wide %value.offset = add i64 %value.row, %time.wide %value.index = add i64 %v.base, %value.offset
%value.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %value.index
%value = load double, ptr addrspace(1) %value.pointer, align 8
%read.decayed = call double @recipe.mul(double %decay, double %read.sum)
%error = call double @recipe.sub(double %value, double %read.decayed) %write.error = call double @recipe.mul(double %write, double %error)
br label %write.loop
write.loop: %write.i = phi i32 [ 0, %read.done ], [ %write.next, %write.step ]
%write.sum = phi double [ 0.0, %read.done ], [ %write.sum.next, %write.step ]
%write.more = icmp ult i32 %write.i, %kwidth br i1 %write.more, label %write.step, label %write.done
write.step: %write.i.wide = zext i32 %write.i to i64 %write.row = mul i64 %write.i.wide, %vwidth.wide %write.cell = add i64 %write.row, %column.wide %write.cell.index = add i64 %work.base, %write.cell
%write.state.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %write.cell.index
%write.state = load double, ptr addrspace(1) %write.state.pointer, align 8
%write.decayed = call double @recipe.mul(double %decay, double %write.state)
%write.offset.row = mul i64 %write.i.wide, %length.wide %write.offset = add i64 %write.offset.row, %time.wide
%write.key.index = add i64 %k.base, %write.offset
%write.key.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %write.key.index
%write.key = load double, ptr addrspace(1) %write.key.pointer, align 8
%write.term = call double @recipe.mul(double %write.key, double %write.error)
%write.state.next = call double @recipe.add(double %write.decayed, double %write.term)
store double %write.state.next, ptr addrspace(1) %write.state.pointer, align 8
%write.query.index = add i64 %q.base, %write.offset
%write.query.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %write.query.index
%write.query = load double, ptr addrspace(1) %write.query.pointer, align 8
%write.output = call double @recipe.mul(double %write.query, double %write.state.next)
%write.sum.next = call double @recipe.add(double %write.sum, double %write.output) %write.next = add nuw i32 %write.i, 1 br label %write.loop
write.done: br i1 %store, label %write.store, label %exit
write.store: %output.index = add i64 %o.base, %value.offset
%output.pointer = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.index
%write.scaled = call double @recipe.mul(double %write.sum, double %scale)
store double %write.scaled, ptr addrspace(1) %output.pointer, align 8 br label %exit
exit: ret void }
define internal void @delta_step( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base, i64 %work.base,
i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, double %decay.scale, i1 %store ) #3 { entry:
%time.wide = zext i32 %time to i64 %kwidth.wide = zext i32 %kwidth to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64
%decay.index = add i64 %a.base, %time.wide
%decay.pointer = getelementptr inbounds double, ptr addrspace(1) %gates, i64 %decay.index
%decay.input = load double, ptr addrspace(1) %decay.pointer, align 8 %softplus = call double @softplus(double %decay.input)
%exponent = call double @recipe.mul(double %softplus, double %decay.scale) %negated = call double @recipe.neg(double %exponent)
%decay = call double @recipe.exp(double %negated) %write.index = add i64 %b.base, %time.wide
%write.pointer = getelementptr inbounds double, ptr addrspace(1) %gates, i64 %write.index
%write.input = load double, ptr addrspace(1) %write.pointer, align 8 %write = call double @sigmoid(double %write.input)
br label %column.loop
column.loop: %column = phi i32 [ 0, %entry ], [ %column.next, %column.done ] %column.more = icmp ult i32 %column, %vwidth
br i1 %column.more, label %column.step, label %exit
column.step: call void @delta_column( ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %k.base, i64 %v.base, i64 %q.base, i64 %o.base, i64 %work.base, i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, i32 %column, double %decay, double %write, i1 %store, double 1.0 )
br label %column.done
column.done: %column.next = add nuw i32 %column, 1 br label %column.loop
exit: ret void }
; One row, head and value column of the gated delta rule at inference: the
; live state carries across windows, starts at zero at position 0, and walks
; only the window's positions.
define internal void @delta_live_body( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %p, i32 %kheads, i32 %kwidth, i32 %vheads, i32 %vwidth, i32 %length, i32 %pairs, i32 %begin, i32 %end, i32 %decode, i1 %tiled, double %scale, i32 %origin ) #3 { entry:
%kheads.wide = zext i32 %kheads to i64 %kwidth.wide = zext i32 %kwidth to i64 %vheads.wide = zext i32 %vheads to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64
%pair = udiv i64 %p, %vwidth.wide %column.wide = urem i64 %p, %vwidth.wide %column = trunc i64 %column.wide to i32
%row = udiv i64 %pair, %vheads.wide %head = urem i64 %pair, %vheads.wide %state = mul i64 %kwidth.wide, %vwidth.wide
%kchannels = mul i64 %kheads.wide, %kwidth.wide %kstream = mul i64 %kchannels, %length.wide
%vchannels = mul i64 %vheads.wide, %vwidth.wide %stream = mul i64 %vchannels, %length.wide
%kplanes = mul i64 %kstream, 2 %row.stride = add i64 %kplanes, %stream
%input.row = mul i64 %row, %row.stride %group = udiv i32 %vheads, %kheads %group.wide = zext i32 %group to i64 %khead.grouped = udiv i64 %head, %group.wide
; A tiled layout gives value head h the key head h modulo the key heads.
%khead.tiled = urem i64 %head, %kheads.wide %khead = select i1 %tiled, i64 %khead.tiled, i64 %khead.grouped
%khead.base = mul i64 %khead, %kwidth.wide %khead.offset = mul i64 %khead.base, %length.wide
%head.base = mul i64 %head, %vwidth.wide %head.offset = mul i64 %head.base, %length.wide
%q.base = add i64 %input.row, %khead.offset %k.base = add i64 %q.base, %kstream
%value.plane = add i64 %input.row, %kplanes %v.base = add i64 %value.plane, %head.offset
%output.row = mul i64 %row, %stream %o.base = add i64 %output.row, %head.offset
%gate.stream = mul i64 %vheads.wide, %length.wide %gate.row = mul i64 %row, %gate.stream %gate.pair = mul i64 %gate.row, 2
%head.length = mul i64 %head, %length.wide %a.base = add i64 %gate.pair, %head.length %b.base = add i64 %a.base, %gate.stream
%work.base = mul i64 %pair, %state
%decay.parameter = call double @recipe.model.weight(ptr addrspace(1) %weights, i64 %head, i32 %decode) %decay.scale = call double @recipe.exp(double %decay.parameter)
%fresh = icmp eq i32 %begin, 0
br i1 %fresh, label %zero.loop, label %time.loop
zero.loop: %zero.i = phi i32 [ 0, %entry ], [ %zero.next, %zero.step ] %zero.more = icmp ult i32 %zero.i, %kwidth
br i1 %zero.more, label %zero.step, label %time.loop
zero.step: %zero.i.wide = zext i32 %zero.i to i64 %zero.row = mul i64 %zero.i.wide, %vwidth.wide %zero.cell = add i64 %zero.row, %column.wide %zero.index = add i64 %work.base, %zero.cell
%zero.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %zero.index
store double 0.0, ptr addrspace(1) %zero.pointer, align 8 %zero.next = add nuw i32 %zero.i, 1 br label %zero.loop
time.loop: %time = phi i32 [ %begin, %entry ], [ %begin, %zero.loop ], [ %time.next, %time.step ] %time.more = icmp ult i32 %time, %end
br i1 %time.more, label %time.step, label %exit
time.step:
; Positions %begin to %end sit at %origin less in a buffer that holds the window alone.
%time.local = sub i32 %time, %origin %time.wide = zext i32 %time.local to i64
%decay.index = add i64 %a.base, %time.wide
%decay.pointer = getelementptr inbounds double, ptr addrspace(1) %gates, i64 %decay.index
%decay.input = load double, ptr addrspace(1) %decay.pointer, align 8 %softplus = call double @softplus(double %decay.input)
%exponent = call double @recipe.mul(double %softplus, double %decay.scale) %negated = call double @recipe.neg(double %exponent)
%decay = call double @recipe.exp(double %negated) %write.index = add i64 %b.base, %time.wide
%write.pointer = getelementptr inbounds double, ptr addrspace(1) %gates, i64 %write.index
%write.input = load double, ptr addrspace(1) %write.pointer, align 8 %write = call double @sigmoid(double %write.input)
call void @delta_column( ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %k.base, i64 %v.base, i64 %q.base, i64 %o.base, i64 %work.base, i32 %kwidth, i32 %vwidth, i32 %length, i32 %time.local, i32 %column, double %decay, double %write, i1 true, double %scale )
%time.next = add nuw i32 %time, 1 br label %time.loop
exit: ret void }
; One row and head of the gated delta rule. The sequence walks in chunks of
; %chunk positions and the carried state is committed at every chunk start, so a
; chunk of one commits each decode step. The chunk never reaches the arithmetic.
define internal void @delta_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %p, i32 %kheads, i32 %kwidth, i32 %vheads, i32 %vwidth, i32 %length, i32 %chunk, i32 %chunks, i32 %pairs, i32 %entries, i32 %decode ) #3 { entry:
%kheads.wide = zext i32 %kheads to i64 %kwidth.wide = zext i32 %kwidth to i64 %vheads.wide = zext i32 %vheads to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64 %chunk.wide = zext i32 %chunk to i64 %chunks.wide = zext i32 %chunks to i64 %pairs.wide = zext i32 %pairs to i64 %entries.wide = zext i32 %entries to i64
%row = udiv i64 %p, %vheads.wide %head = urem i64 %p, %vheads.wide %state = mul i64 %kwidth.wide, %vwidth.wide
%state.i32 = trunc i64 %state to i32 %committing = icmp ne i32 %entries, 0 %commit.count = select i1 %committing, i32 %state.i32, i32 0
%kchannels = mul i64 %kheads.wide, %kwidth.wide %kstream = mul i64 %kchannels, %length.wide
%vchannels = mul i64 %vheads.wide, %vwidth.wide %stream = mul i64 %vchannels, %length.wide
%kplanes = mul i64 %kstream, 2 %row.stride = add i64 %kplanes, %stream
%input.row = mul i64 %row, %row.stride %group = udiv i32 %vheads, %kheads %group.wide = zext i32 %group to i64 %khead = udiv i64 %head, %group.wide
%khead.base = mul i64 %khead, %kwidth.wide %khead.offset = mul i64 %khead.base, %length.wide
%head.base = mul i64 %head, %vwidth.wide %head.offset = mul i64 %head.base, %length.wide
%q.base = add i64 %input.row, %khead.offset %k.base = add i64 %q.base, %kstream
%value.plane = add i64 %input.row, %kplanes %v.base = add i64 %value.plane, %head.offset
%output.row = mul i64 %row, %stream %o.base = add i64 %output.row, %head.offset
%gate.stream = mul i64 %vheads.wide, %length.wide %gate.row = mul i64 %row, %gate.stream %gate.pair = mul i64 %gate.row, 2
%head.length = mul i64 %head, %length.wide %a.base = add i64 %gate.pair, %head.length %b.base = add i64 %a.base, %gate.stream
%entry.span = mul i64 %entries.wide, %state %entry.base = mul i64 %p, %entry.span
%work.region = mul i64 %pairs.wide, %entry.span %work.offset = mul i64 %p, %state %work.base = add i64 %work.region, %work.offset
%decay.parameter = call double @recipe.model.weight(ptr addrspace(1) %weights, i64 %head, i32 %decode) %decay.scale = call double @recipe.exp(double %decay.parameter)
br label %zero.loop
zero.loop: %zero.i = phi i32 [ 0, %entry ], [ %zero.next, %zero.step ] %zero.more = icmp ult i32 %zero.i, %state.i32
br i1 %zero.more, label %zero.step, label %chunk.loop
zero.step: %zero.i.wide = zext i32 %zero.i to i64 %zero.index = add i64 %work.base, %zero.i.wide
%zero.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %zero.index
store double 0.0, ptr addrspace(1) %zero.pointer, align 8 %zero.next = add nuw i32 %zero.i, 1 br label %zero.loop
chunk.loop: %chunk.index = phi i32 [ 0, %zero.loop ], [ %chunk.next, %chunk.done ] %chunk.more = icmp ult i32 %chunk.index, %chunks
%chunk.start = mul i32 %chunk.index, %chunk %chunk.index.wide = zext i32 %chunk.index to i64 %chunk.entry = mul i64 %chunk.index.wide, %state %chunk.entry.base = add i64 %entry.base, %chunk.entry
br i1 %chunk.more, label %commit.loop, label %exit
commit.loop: %commit.i = phi i32 [ 0, %chunk.loop ], [ %commit.next, %commit.step ] %commit.more = icmp ult i32 %commit.i, %commit.count
br i1 %commit.more, label %commit.step, label %time.loop
commit.step: %commit.i.wide = zext i32 %commit.i to i64 %commit.work = add i64 %work.base, %commit.i.wide
%commit.work.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %commit.work
%commit.value = load double, ptr addrspace(1) %commit.work.pointer, align 8 %commit.entry = add i64 %chunk.entry.base, %commit.i.wide
%commit.entry.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %commit.entry
store double %commit.value, ptr addrspace(1) %commit.entry.pointer, align 8 %commit.next = add nuw i32 %commit.i, 1 br label %commit.loop
time.loop: %offset = phi i32 [ 0, %commit.loop ], [ %offset.next, %step.done ] %time = add i32 %chunk.start, %offset
%offset.more = icmp ult i32 %offset, %chunk %time.more = icmp ult i32 %time, %length %step.more = and i1 %offset.more, %time.more
br i1 %step.more, label %step, label %chunk.done
step: call void @delta_step( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %output, ptr addrspace(1) %context,
i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base, i64 %work.base,
i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, double %decay.scale, i1 true )
br label %step.done
step.done: %offset.next = add nuw i32 %offset, 1 br label %time.loop
chunk.done: %chunk.next = add nuw i32 %chunk.index, 1 br label %chunk.loop
exit: ret void }
; One position of the gated delta rule in reverse. %previous indexes the state
; before the position and the state adjoint at %adjoint.base is carried
; backward. The two head vectors at %vector.base hold the readout error and the
; key adjoint weight in that order. The return value is this position's
; contribution to the decay scale gradient.
define internal RECIPE_STATE @delta_back( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %context, ptr addrspace(1) %backward, ptr addrspace(1) %delta,
ptr addrspace(1) %input.adjoint, ptr addrspace(1) %gate.adjoint, i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base,
i64 %previous, i64 %adjoint.base, i64 %vector.base, i32 %kwidth, i32 %vwidth, i32 %length, i32 %time, double %decay.scale ) #3 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%time.wide = zext i32 %time to i64 %kwidth.wide = zext i32 %kwidth to i64 %vwidth.wide = zext i32 %vwidth to i64 %length.wide = zext i32 %length to i64
%decay.index = add i64 %a.base, %time.wide
%decay.pointer = getelementptr inbounds double, ptr addrspace(1) %gates, i64 %decay.index
%decay.input = load double, ptr addrspace(1) %decay.pointer, align 8 %softplus = call double @softplus(double %decay.input)
%exponent = call double @recipe.mul(double %softplus, double %decay.scale) %negated = call double @recipe.neg(double %exponent)
%decay.model = call double @recipe.exp(double %negated) %decay = call RECIPE_STATE @recipe.decode(double %decay.model) %write.index = add i64 %b.base, %time.wide
%write.pointer = getelementptr inbounds double, ptr addrspace(1) %gates, i64 %write.index
%write.input = load double, ptr addrspace(1) %write.pointer, align 8 %write.model = call double @sigmoid(double %write.input) %write = call RECIPE_STATE @recipe.decode(double %write.model)
%weight.base = add i64 %vector.base, %vwidth.wide
br label %seed.row
seed.row: %seed.i = phi i32 [ 0, %entry ], [ %seed.i.next, %seed.row.done ] %seed.i.more = icmp ult i32 %seed.i, %kwidth
%seed.i.wide = zext i32 %seed.i to i64 %seed.i.row = mul i64 %seed.i.wide, %length.wide %seed.i.offset = add i64 %seed.i.row, %time.wide %seed.query.index = add i64 %q.base, %seed.i.offset
br i1 %seed.i.more, label %seed.column, label %column.loop
seed.column: %seed.j = phi i32 [ 0, %seed.row ], [ %seed.j.next, %seed.step ] %seed.j.more = icmp ult i32 %seed.j, %vwidth
br i1 %seed.j.more, label %seed.step, label %seed.row.done
seed.step: %seed.query.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %seed.query.index
%seed.query.model = load double, ptr addrspace(1) %seed.query.pointer, align 8 %seed.query = call RECIPE_STATE @recipe.decode(double %seed.query.model)
%seed.j.wide = zext i32 %seed.j to i64 %seed.j.row = mul i64 %seed.j.wide, %length.wide %seed.j.offset = add i64 %seed.j.row, %time.wide %seed.delta.index = add i64 %o.base, %seed.j.offset
%seed.delta.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %seed.delta.index
%seed.delta = load RECIPE_STATE, ptr addrspace(1) %seed.delta.pointer, align RECIPE_STATE_ALIGN
%seed.product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %seed.query, RECIPE_STATE %seed.delta)
%seed.row.base = mul i64 %seed.i.wide, %vwidth.wide %seed.cell = add i64 %seed.row.base, %seed.j.wide %seed.index = add i64 %adjoint.base, %seed.cell
%seed.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %seed.index
%seed.prior = load RECIPE_STATE, ptr addrspace(1) %seed.pointer, align RECIPE_STATE_ALIGN
%seed.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %seed.prior, RECIPE_STATE %seed.product)
store RECIPE_STATE %seed.total, ptr addrspace(1) %seed.pointer, align RECIPE_STATE_ALIGN %seed.j.next = add nuw i32 %seed.j, 1 br label %seed.column
seed.row.done: %seed.i.next = add nuw i32 %seed.i, 1 br label %seed.row
column.loop: %column = phi i32 [ 0, %seed.row ], [ %column.next, %column.done ]
%write.gradient = phi RECIPE_STATE [ %state.zero, %seed.row ], [ %write.gradient.next, %column.done ]
%column.wide = zext i32 %column to i64
%column.more = icmp ult i32 %column, %vwidth br i1 %column.more, label %column.row, label %row.loop
column.row: %column.i = phi i32 [ 0, %column.loop ], [ %column.i.next, %column.step ]
%readout = phi RECIPE_STATE [ %state.zero, %column.loop ], [ %readout.next, %column.step ]
%weight = phi RECIPE_STATE [ %state.zero, %column.loop ], [ %weight.next, %column.step ]
%column.i.more = icmp ult i32 %column.i, %kwidth br i1 %column.i.more, label %column.step, label %column.store
column.step: %column.i.wide = zext i32 %column.i to i64 %column.i.row = mul i64 %column.i.wide, %vwidth.wide %column.cell = add i64 %column.i.row, %column.wide
%column.state.index = add i64 %previous, %column.cell
%column.state.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %column.state.index
%column.state.model = load double, ptr addrspace(1) %column.state.pointer, align 8 %column.state = call RECIPE_STATE @recipe.decode(double %column.state.model)
%column.adjoint.index = add i64 %adjoint.base, %column.cell
%column.adjoint.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %column.adjoint.index
%column.adjoint = load RECIPE_STATE, ptr addrspace(1) %column.adjoint.pointer, align RECIPE_STATE_ALIGN
%column.i.offset.row = mul i64 %column.i.wide, %length.wide %column.i.offset = add i64 %column.i.offset.row, %time.wide
%column.key.index = add i64 %k.base, %column.i.offset
%column.key.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %column.key.index
%column.key.model = load double, ptr addrspace(1) %column.key.pointer, align 8 %column.key = call RECIPE_STATE @recipe.decode(double %column.key.model)
%column.readout = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %column.key, RECIPE_STATE %column.state)
%readout.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %readout, RECIPE_STATE %column.readout)
%column.weight = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %column.key, RECIPE_STATE %column.adjoint)
%weight.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %weight, RECIPE_STATE %column.weight)
%column.i.next = add nuw i32 %column.i, 1 br label %column.row
column.store: %column.row.offset = mul i64 %column.wide, %length.wide %column.offset = add i64 %column.row.offset, %time.wide
%column.value.index = add i64 %v.base, %column.offset
%column.value.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %column.value.index
%column.value.model = load double, ptr addrspace(1) %column.value.pointer, align 8 %column.value = call RECIPE_STATE @recipe.decode(double %column.value.model)
%readout.decayed = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay, RECIPE_STATE %readout)
%error = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %column.value, RECIPE_STATE %readout.decayed)
%error.index = add i64 %vector.base, %column.wide
%error.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %error.index
store RECIPE_STATE %error, ptr addrspace(1) %error.pointer, align RECIPE_STATE_ALIGN
%weight.index = add i64 %weight.base, %column.wide
%weight.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %weight.index
store RECIPE_STATE %weight, ptr addrspace(1) %weight.pointer, align RECIPE_STATE_ALIGN
%value.gradient = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %write, RECIPE_STATE %weight)
%value.adjoint.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %input.adjoint, i64 %column.value.index
%value.prior = load RECIPE_STATE, ptr addrspace(1) %value.adjoint.pointer, align RECIPE_STATE_ALIGN
%value.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %value.prior, RECIPE_STATE %value.gradient)
store RECIPE_STATE %value.total, ptr addrspace(1) %value.adjoint.pointer, align RECIPE_STATE_ALIGN
%write.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %error, RECIPE_STATE %weight)
%write.gradient.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %write.gradient, RECIPE_STATE %write.term) br label %column.done
column.done: %column.next = add nuw i32 %column, 1 br label %column.loop
row.loop: %row.i = phi i32 [ 0, %column.loop ], [ %row.i.next, %row.done ]
%decay.gradient = phi RECIPE_STATE [ %state.zero, %column.loop ], [ %decay.gradient.next, %row.done ]
%row.i.more = icmp ult i32 %row.i, %kwidth
%row.i.wide = zext i32 %row.i to i64 %row.i.offset.row = mul i64 %row.i.wide, %length.wide %row.i.offset = add i64 %row.i.offset.row, %time.wide
%row.key.index = add i64 %k.base, %row.i.offset %row.query.index = add i64 %q.base, %row.i.offset %row.i.base = mul i64 %row.i.wide, %vwidth.wide
br i1 %row.i.more, label %row.column, label %gates.entry
row.column: %row.j = phi i32 [ 0, %row.loop ], [ %row.j.next, %row.step ]
%decay.part = phi RECIPE_STATE [ %state.zero, %row.loop ], [ %decay.part.next, %row.step ]
%key.direct = phi RECIPE_STATE [ %state.zero, %row.loop ], [ %key.direct.next, %row.step ]
%key.readout = phi RECIPE_STATE [ %state.zero, %row.loop ], [ %key.readout.next, %row.step ]
%query.part = phi RECIPE_STATE [ %state.zero, %row.loop ], [ %query.part.next, %row.step ]
%row.j.more = icmp ult i32 %row.j, %vwidth br i1 %row.j.more, label %row.step, label %row.store
row.step: %row.j.wide = zext i32 %row.j to i64 %row.cell = add i64 %row.i.base, %row.j.wide %row.state.index = add i64 %previous, %row.cell
%row.state.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %row.state.index
%row.state.model = load double, ptr addrspace(1) %row.state.pointer, align 8 %row.state = call RECIPE_STATE @recipe.decode(double %row.state.model)
%row.adjoint.index = add i64 %adjoint.base, %row.cell
%row.adjoint.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %row.adjoint.index
%row.adjoint = load RECIPE_STATE, ptr addrspace(1) %row.adjoint.pointer, align RECIPE_STATE_ALIGN
%row.error.index = add i64 %vector.base, %row.j.wide
%row.error.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %row.error.index
%row.error = load RECIPE_STATE, ptr addrspace(1) %row.error.pointer, align RECIPE_STATE_ALIGN
%row.weight.index = add i64 %weight.base, %row.j.wide
%row.weight.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %row.weight.index
%row.weight = load RECIPE_STATE, ptr addrspace(1) %row.weight.pointer, align RECIPE_STATE_ALIGN
%row.key.pointer = getelementptr inbounds double, ptr addrspace(1) %input, i64 %row.key.index
%row.key.model = load double, ptr addrspace(1) %row.key.pointer, align 8 %row.key = call RECIPE_STATE @recipe.decode(double %row.key.model)
%row.write.weight = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %write, RECIPE_STATE %row.weight)
%row.adjoint.written = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %row.key, RECIPE_STATE %row.write.weight)
%row.adjoint.kept = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %row.adjoint, RECIPE_STATE %row.adjoint.written)
%row.decay.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %row.state, RECIPE_STATE %row.adjoint.kept)
%decay.part.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %decay.part, RECIPE_STATE %row.decay.term)
%row.direct.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %row.error, RECIPE_STATE %row.adjoint)
%key.direct.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %key.direct, RECIPE_STATE %row.direct.term)
%row.readout.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %row.weight, RECIPE_STATE %row.state)
%key.readout.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %key.readout, RECIPE_STATE %row.readout.term)
%row.state.decayed = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay, RECIPE_STATE %row.state)
%row.write.error = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %write, RECIPE_STATE %row.error)
%row.state.written = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %row.key, RECIPE_STATE %row.write.error)
%row.state.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %row.state.decayed, RECIPE_STATE %row.state.written)
%row.j.offset.row = mul i64 %row.j.wide, %length.wide %row.j.offset = add i64 %row.j.offset.row, %time.wide
%row.delta.index = add i64 %o.base, %row.j.offset
%row.delta.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %row.delta.index
%row.delta = load RECIPE_STATE, ptr addrspace(1) %row.delta.pointer, align RECIPE_STATE_ALIGN
%row.query.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %row.delta, RECIPE_STATE %row.state.next)
%query.part.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %query.part, RECIPE_STATE %row.query.term)
%row.adjoint.next = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay, RECIPE_STATE %row.adjoint.kept)
store RECIPE_STATE %row.adjoint.next, ptr addrspace(1) %row.adjoint.pointer, align RECIPE_STATE_ALIGN
%row.j.next = add nuw i32 %row.j, 1 br label %row.column
row.store: %key.readout.decayed = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay, RECIPE_STATE %key.readout)
%key.difference = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %key.direct, RECIPE_STATE %key.readout.decayed)
%key.gradient = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %write, RECIPE_STATE %key.difference)
%key.adjoint.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %input.adjoint, i64 %row.key.index
%key.prior = load RECIPE_STATE, ptr addrspace(1) %key.adjoint.pointer, align RECIPE_STATE_ALIGN
%key.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %key.prior, RECIPE_STATE %key.gradient)
store RECIPE_STATE %key.total, ptr addrspace(1) %key.adjoint.pointer, align RECIPE_STATE_ALIGN
%query.adjoint.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %input.adjoint, i64 %row.query.index
%query.prior = load RECIPE_STATE, ptr addrspace(1) %query.adjoint.pointer, align RECIPE_STATE_ALIGN
%query.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %query.prior, RECIPE_STATE %query.part)
store RECIPE_STATE %query.total, ptr addrspace(1) %query.adjoint.pointer, align RECIPE_STATE_ALIGN
%decay.gradient.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %decay.gradient, RECIPE_STATE %decay.part) br label %row.done
row.done: %row.i.next = add nuw i32 %row.i, 1 br label %row.loop
gates.entry: %decay.slope.model = call double @sigmoid(double %decay.input) %decay.slope = call RECIPE_STATE @recipe.decode(double %decay.slope.model) %decay.scale.wide = call RECIPE_STATE @recipe.decode(double %decay.scale) %decay.factor = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay.scale.wide, RECIPE_STATE %decay)
%decay.chain = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay.gradient, RECIPE_STATE %decay.factor)
%decay.chain.negated = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %decay.chain)
%decay.input.gradient = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay.chain.negated, RECIPE_STATE %decay.slope)
%decay.adjoint.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gate.adjoint, i64 %decay.index
%decay.prior = load RECIPE_STATE, ptr addrspace(1) %decay.adjoint.pointer, align RECIPE_STATE_ALIGN
%decay.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %decay.prior, RECIPE_STATE %decay.input.gradient)
store RECIPE_STATE %decay.total, ptr addrspace(1) %decay.adjoint.pointer, align RECIPE_STATE_ALIGN
%write.complement = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %write) %write.slope = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %write, RECIPE_STATE %write.complement)
%write.input.gradient = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %write.gradient, RECIPE_STATE %write.slope)
%write.adjoint.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gate.adjoint, i64 %write.index
%write.prior = load RECIPE_STATE, ptr addrspace(1) %write.adjoint.pointer, align RECIPE_STATE_ALIGN
%write.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %write.prior, RECIPE_STATE %write.input.gradient)
store RECIPE_STATE %write.total, ptr addrspace(1) %write.adjoint.pointer, align RECIPE_STATE_ALIGN
%softplus.wide = call RECIPE_STATE @recipe.decode(double %softplus) %scale.chain = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %softplus.wide, RECIPE_STATE %decay.factor) %scale.negated = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %scale.chain)
%scale.gradient = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %decay.gradient, RECIPE_STATE %scale.negated) ret RECIPE_STATE %scale.gradient }
; One row and head of the gated delta rule in reverse. Each chunk is replayed
; forward from its committed entry state so the state before every position is
; available, then the chunk is walked backward. The decay scale gradient lands
; in this pair's partial for the fold below.
define internal void @delta_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %weights, ptr addrspace(1) %context, ptr addrspace(1) %backward,
ptr addrspace(1) %delta, ptr addrspace(1) %input.adjoint, ptr addrspace(1) %gate.adjoint,
i64 %p, i32 %kheads, i32 %kwidth, i32 %vheads, i32 %vwidth, i32 %length, i32 %chunk, i32 %chunks, i32 %pairs ) #3 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
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
%khead.first = mul i64 %khead, %group.wide %pair.row = mul i64 %row, %vheads.wide
br label %head.loop
; Every value head sharing this key head walks in turn, so one thread owns the
; query and key adjoint elements of the head they share. One value head per key
; head makes exactly one pass and keeps the ungrouped indexing.
head.loop: %g = phi i32 [ 0, %entry ], [ %g.next, %head.done ] %g.more = icmp ult i32 %g, %group
br i1 %g.more, label %head.body, label %exit
head.body: %g.wide = zext i32 %g to i64 %head = add i64 %khead.first, %g.wide %pair = add i64 %pair.row, %head
%head.base = mul i64 %head, %vwidth.wide %head.offset = mul i64 %head.base, %length.wide
%v.base = add i64 %value.plane, %head.offset %o.base = add i64 %output.row, %head.offset
%head.length = mul i64 %head, %length.wide %a.base = add i64 %gate.pair, %head.length %b.base = add i64 %a.base, %gate.stream
%entry.base = mul i64 %pair, %entry.span
%work.offset = mul i64 %pair, %state %work.base = add i64 %work.region, %work.offset
%replay.offset = mul i64 %pair, %replay.span %replay.base = add i64 %replay.region, %replay.offset
%adjoint.base = add i64 %adjoint.region, %work.offset
%vector.offset = mul i64 %pair, %vector.span %vector.base = add i64 %vector.region, %vector.offset
%partial.index = add i64 %partial.region, %pair
%head.i32 = trunc i64 %head to i32 %decay.pointer = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %head
%decay.parameter = load double, ptr addrspace(1) %decay.pointer, align 8 %decay.scale = call double @recipe.exp(double %decay.parameter)
br label %zero.loop
zero.loop: %zero.i = phi i32 [ 0, %head.body ], [ %zero.next, %zero.step ] %zero.more = icmp ult i32 %zero.i, %state.i32
br i1 %zero.more, label %zero.step, label %chunk.loop
zero.step: %zero.i.wide = zext i32 %zero.i to i64 %zero.index = add i64 %adjoint.base, %zero.i.wide
%zero.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %zero.index
store RECIPE_STATE %state.zero, ptr addrspace(1) %zero.pointer, align RECIPE_STATE_ALIGN %zero.next = add nuw i32 %zero.i, 1 br label %zero.loop
chunk.loop: %remaining = phi i32 [ %chunks, %zero.loop ], [ %remaining.next, %chunk.done ]
%total = phi RECIPE_STATE [ %state.zero, %zero.loop ], [ %decay.sum, %chunk.done ]
%chunk.more = icmp ugt i32 %remaining, 0 %chunk.index = sub i32 %remaining, 1
%chunk.start = mul i32 %chunk.index, %chunk %chunk.index.wide = zext i32 %chunk.index to i64 %chunk.entry = mul i64 %chunk.index.wide, %state %chunk.entry.base = add i64 %entry.base, %chunk.entry
br i1 %chunk.more, label %restore.loop, label %store
restore.loop: %restore.i = phi i32 [ 0, %chunk.loop ], [ %restore.next, %restore.step ] %restore.more = icmp ult i32 %restore.i, %state.i32
br i1 %restore.more, label %restore.step, label %replay.loop
restore.step: %restore.i.wide = zext i32 %restore.i to i64 %restore.entry = add i64 %chunk.entry.base, %restore.i.wide
%restore.entry.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %restore.entry
%restore.value = load double, ptr addrspace(1) %restore.entry.pointer, align 8 %restore.work = add i64 %work.base, %restore.i.wide
%restore.work.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %restore.work
store double %restore.value, ptr addrspace(1) %restore.work.pointer, align 8 %restore.next = add nuw i32 %restore.i, 1 br label %restore.loop
replay.loop: %replay.i = phi i32 [ 0, %restore.loop ], [ %replay.i.next, %replay.step ] %replay.time = add i32 %chunk.start, %replay.i
%replay.i.more = icmp ult i32 %replay.i, %chunk %replay.time.more = icmp ult i32 %replay.time, %length
%replay.more = and i1 %replay.i.more, %replay.time.more
%replay.i.wide = zext i32 %replay.i to i64 %replay.slot.offset = mul i64 %replay.i.wide, %state %replay.slot = add i64 %replay.base, %replay.slot.offset
br i1 %replay.more, label %replay.save, label %backward.loop
replay.save: %save.i = phi i32 [ 0, %replay.loop ], [ %save.next, %save.step ] %save.more = icmp ult i32 %save.i, %state.i32
br i1 %save.more, label %save.step, label %replay.step
save.step: %save.i.wide = zext i32 %save.i to i64 %save.work = add i64 %work.base, %save.i.wide
%save.work.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %save.work
%save.value = load double, ptr addrspace(1) %save.work.pointer, align 8 %save.slot = add i64 %replay.slot, %save.i.wide
%save.slot.pointer = getelementptr inbounds double, ptr addrspace(1) %context, i64 %save.slot
store double %save.value, ptr addrspace(1) %save.slot.pointer, align 8 %save.next = add nuw i32 %save.i, 1 br label %replay.save
replay.step: call void @delta_step( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %context, ptr addrspace(1) %context,
i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base, i64 %work.base,
i32 %kwidth, i32 %vwidth, i32 %length, i32 %replay.time, double %decay.scale, i1 false )
%replay.i.next = add nuw i32 %replay.i, 1 br label %replay.loop
backward.loop: %backward.i = phi i32 [ %replay.i, %replay.loop ], [ %backward.index, %backward.step ]
%decay.sum = phi RECIPE_STATE [ %total, %replay.loop ], [ %decay.sum.next, %backward.step ]
%backward.more = icmp ugt i32 %backward.i, 0 br i1 %backward.more, label %backward.step, label %chunk.done
backward.step: %backward.index = sub i32 %backward.i, 1 %backward.time = add i32 %chunk.start, %backward.index %backward.index.wide = zext i32 %backward.index to i64
%backward.slot.offset = mul i64 %backward.index.wide, %state %backward.slot = add i64 %replay.base, %backward.slot.offset
%contribution = call RECIPE_STATE @delta_back( ptr addrspace(1) %input, ptr addrspace(1) %gates, ptr addrspace(1) %context, ptr addrspace(1) %backward, ptr addrspace(1) %delta,
ptr addrspace(1) %input.adjoint, ptr addrspace(1) %gate.adjoint, i64 %q.base, i64 %k.base, i64 %v.base, i64 %o.base, i64 %a.base, i64 %b.base,
i64 %backward.slot, i64 %adjoint.base, i64 %vector.base, i32 %kwidth, i32 %vwidth, i32 %length, i32 %backward.time, double %decay.scale )
%decay.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %decay.sum, RECIPE_STATE %contribution) br label %backward.loop
chunk.done: %remaining.next = sub i32 %remaining, 1 br label %chunk.loop
store: %partial.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %partial.index
store RECIPE_STATE %total, ptr addrspace(1) %partial.pointer, align RECIPE_STATE_ALIGN br label %head.done
head.done: %g.next = add nuw i32 %g, 1 br label %head.loop
exit: ret void }
; Decay scale %p sums its per-row partials in row order and writes the gradient.
define internal void @delta_reverse_decay_body( ptr addrspace(1) %backward, ptr addrspace(1) %gradient, i64 %p, i32 %rows, i32 %heads, i32 %partials, i32 %offset ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%heads.wide = zext i32 %heads to i64 %partials.wide = zext i32 %partials to i64 %offset.wide = zext i32 %offset to i64
br label %loop loop: %r = phi i32 [ 0, %entry ], [ %next, %step ] %sum = phi RECIPE_STATE [ %state.zero, %entry ], [ %sum.next, %step ]
%more = icmp ult i32 %r, %rows br i1 %more, label %step, label %done
step: %r.wide = zext i32 %r to i64 %pair.base = mul i64 %r.wide, %heads.wide %pair = add i64 %pair.base, %p %index = add i64 %partials.wide, %pair
%pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %index
%value = load RECIPE_STATE, ptr addrspace(1) %pointer, align RECIPE_STATE_ALIGN %sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %value)
%next = add nuw i32 %r, 1 br label %loop
done: %gradient.index = add i64 %offset.wide, %p
%gradient.pointer = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gradient, i64 %gradient.index
store RECIPE_STATE %sum, ptr addrspace(1) %gradient.pointer, align RECIPE_STATE_ALIGN ret void }
; Mixture routing bodies. Scores and weights hold %experts values per position,
; expert e of position t at [e * length + t] within the row.
; Position %p keeps its %top highest scores, ties to the lower expert, under a
; softmax over the kept scores, and zero elsewhere. The weights buffer holds
; the selection marks until the last pass overwrites them.
; One expert unnormalized routing score: exp(score - maximum) under a softmax,
; the logistic of the score under a sigmoid.
define internal double @topk_score( double %score, double %maximum, i1 %sigmoid ) #1 { entry:
%shifted = call double @recipe.sub(double %score, double %maximum) %exponential = call double @recipe.exp(double %shifted)
%logistic = call double @sigmoid(double %score) %result = select i1 %sigmoid, double %logistic, double %exponential ret double %result }
; The router of one position on one wave: each lane holds the experts
; lane, lane + width, ..., and the wave picks the top scores one at a time by
; a wave maximum that prefers the lower expert on a tie, as the serial scan
; does. The maximum, the total and the written weights follow the serial body.
define internal void @topk_forward_wave_body( ptr addrspace(1) %scores, ptr addrspace(1) %weights, i64 %p, i32 %experts, i32 %length, i32 %top, i32 %scoring, i32 %renormalize, i32 %lane, i32 %width ) #1 { entry:
%length.wide = zext i32 %length to i64
%sigmoid = icmp ne i32 %scoring, 0 %renorm = icmp ne i32 %renormalize, 0 %every = xor i1 %renorm, true %plain = xor i1 %sigmoid, true %divide = or i1 %renorm, %plain
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%lanes.over = add i32 %experts, %width %lanes.raised = sub i32 %lanes.over, 1 %slots = udiv i32 %lanes.raised, %width
%reduce.start = udiv i32 %width, 2
br label %pick.loop
pick.loop:
%pick = phi i32 [ 0, %entry ], [ %pick.next, %pick.mark ]
%mask = phi i64 [ 0, %entry ], [ %mask.next, %pick.mark ]
%pick.more = icmp ult i32 %pick, %top
br i1 %pick.more, label %own.loop, label %max.entry
own.loop:
%own.j = phi i32 [ 0, %pick.loop ], [ %own.j.next, %own.step ]
%own.best = phi i32 [ -1, %pick.loop ], [ %own.best.next, %own.step ]
%own.score = phi RECIPE_STATE [ %state.zero, %pick.loop ], [ %own.score.next, %own.step ]
%own.more = icmp ult i32 %own.j, %slots
br i1 %own.more, label %own.step, label %reduce.entry
own.step:
%own.base = mul i32 %own.j, %width %own.e = add i32 %own.base, %lane
%own.in = icmp ult i32 %own.e, %experts
%own.bit = zext i32 %own.j to i64 %own.flag = shl i64 1, %own.bit %own.taken.bits = and i64 %mask, %own.flag %own.taken = icmp ne i64 %own.taken.bits, 0
%own.free = xor i1 %own.taken, true %own.open = and i1 %own.in, %own.free
%own.e.safe = select i1 %own.in, i32 %own.e, i32 0 %own.e.wide = zext i32 %own.e.safe to i64 %own.offset = mul i64 %own.e.wide, %length.wide %own.index = add i64 %p, %own.offset
%own.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %own.index %own.model = load double, ptr addrspace(1) %own.ptr, align 8
%own.value = call RECIPE_STATE @recipe.decode(double %own.model)
%own.none = icmp slt i32 %own.best, 0 %own.higher = call i1 @recipe.state.ogt(RECIPE_STATE %own.value, RECIPE_STATE %own.score) %own.better = or i1 %own.none, %own.higher %own.take = and i1 %own.open, %own.better
%own.best.next = select i1 %own.take, i32 %own.e, i32 %own.best %own.score.next = select i1 %own.take, RECIPE_STATE %own.value, RECIPE_STATE %own.score
%own.j.next = add i32 %own.j, 1
br label %own.loop
reduce.entry:
br label %reduce.loop
reduce.loop:
%reduce.offset = phi i32 [ %reduce.start, %reduce.entry ], [ %reduce.offset.next, %reduce.step ]
%best = phi i32 [ %own.best, %reduce.entry ], [ %best.next, %reduce.step ]
%best.score = phi RECIPE_STATE [ %own.score, %reduce.entry ], [ %best.score.next, %reduce.step ]
%reduce.more = icmp ugt i32 %reduce.offset, 0
br i1 %reduce.more, label %reduce.step, label %pick.mark
reduce.step:
%partner.lane = xor i32 %lane, %reduce.offset %partner.index = mul i32 %partner.lane, 4
%partner.score = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %best.score, i32 %partner.index)
%best.float = bitcast i32 %best to float
%partner.best.float = call float @recipe.wave.partner.f32(float %best.float, i32 %partner.index)
%partner.best = bitcast float %partner.best.float to i32
%partner.valid = icmp sge i32 %partner.best, 0 %mine.invalid = icmp slt i32 %best, 0
%partner.higher = call i1 @recipe.state.ogt(RECIPE_STATE %partner.score, RECIPE_STATE %best.score)
%partner.equal = call i1 @recipe.state.oeq(RECIPE_STATE %partner.score, RECIPE_STATE %best.score)
%partner.lower = icmp slt i32 %partner.best, %best
%partner.tie = and i1 %partner.equal, %partner.lower
%partner.wins.valid = or i1 %partner.higher, %partner.tie
%partner.wins.any = or i1 %mine.invalid, %partner.wins.valid
%partner.wins = and i1 %partner.valid, %partner.wins.any
%best.next = select i1 %partner.wins, i32 %partner.best, i32 %best
%best.score.next = select i1 %partner.wins, RECIPE_STATE %partner.score, RECIPE_STATE %best.score
%reduce.offset.next = udiv i32 %reduce.offset, 2
br label %reduce.loop
pick.mark:
%mark.lane = urem i32 %best, %width %mark.mine = icmp eq i32 %mark.lane, %lane %mark.valid = icmp sge i32 %best, 0 %mark.set = and i1 %mark.mine, %mark.valid
%mark.slot = udiv i32 %best, %width %mark.bit = zext i32 %mark.slot to i64 %mark.flag = shl i64 1, %mark.bit
%mark.masked = or i64 %mask, %mark.flag
%mask.next = select i1 %mark.set, i64 %mark.masked, i64 %mask
%pick.next = add i32 %pick, 1
br label %pick.loop
max.entry:
br label %max.loop
max.loop:
%m.j = phi i32 [ 0, %max.entry ], [ %m.j.next, %max.step ]
%m.value = phi RECIPE_STATE [ %state.zero, %max.entry ], [ %m.value.next, %max.step ]
%m.first = phi i1 [ true, %max.entry ], [ %m.first.next, %max.step ]
%m.more = icmp ult i32 %m.j, %slots
br i1 %m.more, label %max.step, label %max.reduce.entry
max.step:
%m.base = mul i32 %m.j, %width %m.e = add i32 %m.base, %lane %m.in = icmp ult i32 %m.e, %experts
%m.bit = zext i32 %m.j to i64 %m.flag = shl i64 1, %m.bit %m.marked.bits = and i64 %mask, %m.flag %m.marked = icmp ne i64 %m.marked.bits, 0
%m.kind = or i1 %m.marked, %every %m.member = and i1 %m.kind, %m.in
%m.e.safe = select i1 %m.in, i32 %m.e, i32 0 %m.e.wide = zext i32 %m.e.safe to i64 %m.offset = mul i64 %m.e.wide, %length.wide %m.index = add i64 %p, %m.offset
%m.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %m.index %m.model = load double, ptr addrspace(1) %m.ptr, align 8 %m.score = call RECIPE_STATE @recipe.decode(double %m.model)
%m.higher = call i1 @recipe.state.ogt(RECIPE_STATE %m.score, RECIPE_STATE %m.value) %m.better = or i1 %m.first, %m.higher %m.take = and i1 %m.member, %m.better
%m.value.next = select i1 %m.take, RECIPE_STATE %m.score, RECIPE_STATE %m.value %m.first.next = select i1 %m.take, i1 false, i1 %m.first
%m.j.next = add i32 %m.j, 1
br label %max.loop
max.reduce.entry:
br label %max.reduce.loop
max.reduce.loop:
%mr.offset = phi i32 [ %reduce.start, %max.reduce.entry ], [ %mr.offset.next, %max.reduce.step ]
%mr.value = phi RECIPE_STATE [ %m.value, %max.reduce.entry ], [ %mr.value.next, %max.reduce.step ]
%mr.empty = phi i1 [ %m.first, %max.reduce.entry ], [ %mr.empty.next, %max.reduce.step ]
%mr.more = icmp ugt i32 %mr.offset, 0
br i1 %mr.more, label %max.reduce.step, label %sum.entry
max.reduce.step:
%mr.partner.lane = xor i32 %lane, %mr.offset %mr.partner.index = mul i32 %mr.partner.lane, 4
%mr.partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %mr.value, i32 %mr.partner.index)
%mr.empty.float = select i1 %mr.empty, float 1.0, float 0.0
%mr.partner.empty.float = call float @recipe.wave.partner.f32(float %mr.empty.float, i32 %mr.partner.index)
%mr.partner.empty = fcmp une float %mr.partner.empty.float, 0.0
%mr.partner.full = xor i1 %mr.partner.empty, true
%mr.higher = call i1 @recipe.state.ogt(RECIPE_STATE %mr.partner, RECIPE_STATE %mr.value)
%mr.better = or i1 %mr.empty, %mr.higher
%mr.take = and i1 %mr.partner.full, %mr.better
%mr.value.next = select i1 %mr.take, RECIPE_STATE %mr.partner, RECIPE_STATE %mr.value
%mr.empty.next = and i1 %mr.empty, %mr.partner.empty
%mr.offset.next = udiv i32 %mr.offset, 2
br label %max.reduce.loop
sum.entry:
%maximum.state = select i1 %mr.empty, RECIPE_STATE %state.zero, RECIPE_STATE %mr.value
%maximum = call double @recipe.encode(RECIPE_STATE %maximum.state)
br label %sum.loop
sum.loop:
%s.j = phi i32 [ 0, %sum.entry ], [ %s.j.next, %sum.step ]
%s.total = phi RECIPE_STATE [ %state.zero, %sum.entry ], [ %s.total.next, %sum.step ]
%s.more = icmp ult i32 %s.j, %slots
br i1 %s.more, label %sum.step, label %sum.reduce.entry
sum.step:
%s.base = mul i32 %s.j, %width %s.e = add i32 %s.base, %lane %s.in = icmp ult i32 %s.e, %experts
%s.bit = zext i32 %s.j to i64 %s.flag = shl i64 1, %s.bit %s.marked.bits = and i64 %mask, %s.flag %s.marked = icmp ne i64 %s.marked.bits, 0
%s.kind = or i1 %s.marked, %every %s.member = and i1 %s.kind, %s.in
%s.e.safe = select i1 %s.in, i32 %s.e, i32 0 %s.e.wide = zext i32 %s.e.safe to i64 %s.offset = mul i64 %s.e.wide, %length.wide %s.index = add i64 %p, %s.offset
%s.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %s.index %s.model = load double, ptr addrspace(1) %s.ptr, align 8
%s.raw = call double @topk_score(double %s.model, double %maximum, i1 %sigmoid) %s.raw.state = call RECIPE_STATE @recipe.decode(double %s.raw)
%s.term = select i1 %s.member, RECIPE_STATE %s.raw.state, RECIPE_STATE %state.zero
%s.total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %s.total, RECIPE_STATE %s.term)
%s.j.next = add i32 %s.j, 1
br label %sum.loop
sum.reduce.entry:
br label %sum.reduce.loop
sum.reduce.loop:
%sr.offset = phi i32 [ %reduce.start, %sum.reduce.entry ], [ %sr.offset.next, %sum.reduce.step ]
%sr.total = phi RECIPE_STATE [ %s.total, %sum.reduce.entry ], [ %sr.total.next, %sum.reduce.step ]
%sr.more = icmp ugt i32 %sr.offset, 0
br i1 %sr.more, label %sum.reduce.step, label %write.entry
sum.reduce.step:
%sr.partner.lane = xor i32 %lane, %sr.offset %sr.partner.index = mul i32 %sr.partner.lane, 4
%sr.partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %sr.total, i32 %sr.partner.index)
%sr.total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sr.total, RECIPE_STATE %sr.partner)
%sr.offset.next = udiv i32 %sr.offset, 2
br label %sum.reduce.loop
write.entry:
%total = call double @recipe.encode(RECIPE_STATE %sr.total)
%denominator = select i1 %divide, double %total, double 1.0
br label %write.loop
write.loop:
%w.j = phi i32 [ 0, %write.entry ], [ %w.j.next, %write.next ]
%w.more = icmp ult i32 %w.j, %slots
br i1 %w.more, label %write.step, label %exit
write.step:
%w.base = mul i32 %w.j, %width %w.e = add i32 %w.base, %lane %w.in = icmp ult i32 %w.e, %experts
%w.bit = zext i32 %w.j to i64 %w.flag = shl i64 1, %w.bit %w.marked.bits = and i64 %mask, %w.flag %w.marked = icmp ne i64 %w.marked.bits, 0
br i1 %w.in, label %write.store, label %write.next
write.store:
%w.e.wide = zext i32 %w.e to i64 %w.offset = mul i64 %w.e.wide, %length.wide %w.index = add i64 %p, %w.offset
%w.score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %w.index %w.score = load double, ptr addrspace(1) %w.score.ptr, align 8
%w.raw = call double @topk_score(double %w.score, double %maximum, i1 %sigmoid) %w.probability = call double @recipe.div(double %w.raw, double %denominator)
%w.value = select i1 %w.marked, double %w.probability, double 0.0
%w.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %w.index
store double %w.value, ptr addrspace(1) %w.ptr, align 8
br label %write.next
write.next:
%w.j.next = add i32 %w.j, 1
br label %write.loop
exit: ret void }
; The routing weights of one position. The top scores are kept by rank, scored
; by softmax over every expert or by sigmoid, and divided by the kept total when
; the block renormalizes. A plain softmax divides by every expert instead, which
; is the evaluate-all-then-mask reference; a plain sigmoid divides by nothing.
define internal void @topk_forward_body( ptr addrspace(1) %scores, ptr addrspace(1) %weights, i64 %p, i32 %experts, i32 %length, i32 %top, i32 %scoring, i32 %renormalize ) #1 { entry:
%experts.wide = zext i32 %experts to i64 %length.wide = zext i32 %length to i64 %top.wide = zext i32 %top to i64
%row = udiv i64 %p, %length.wide %position = urem i64 %p, %length.wide %per.row = mul i64 %experts.wide, %length.wide %row.base = mul i64 %row, %per.row %base = add i64 %row.base, %position
%sigmoid = icmp ne i32 %scoring, 0 %renorm = icmp ne i32 %renormalize, 0 %every = xor i1 %renorm, true %plain = xor i1 %sigmoid, true %divide = or i1 %renorm, %plain
br label %clear.loop clear.loop: %clear = phi i64 [ 0, %entry ], [ %clear.next, %clear.step ] %clear.more = icmp ult i64 %clear, %experts.wide
br i1 %clear.more, label %clear.step, label %select.loop clear.step: %clear.offset = mul i64 %clear, %length.wide %clear.index = add i64 %base, %clear.offset
%clear.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %clear.index store double 0.0, ptr addrspace(1) %clear.ptr, align 8 %clear.next = add i64 %clear, 1 br label %clear.loop
select.loop: %pick = phi i64 [ 0, %clear.loop ], [ %pick.next, %select.mark ] %pick.more = icmp ult i64 %pick, %top.wide br i1 %pick.more, label %scan.entry, label %normalize.entry
scan.entry: br label %scan.loop
scan.loop: %candidate = phi i64 [ 0, %scan.entry ], [ %candidate.next, %scan.step ] %best = phi i64 [ -1, %scan.entry ], [ %best.next, %scan.step ]
%best.score = phi double [ 0.0, %scan.entry ], [ %best.score.next, %scan.step ] %scan.more = icmp ult i64 %candidate, %experts.wide br i1 %scan.more, label %scan.step, label %select.mark
scan.step: %candidate.offset = mul i64 %candidate, %length.wide %candidate.index = add i64 %base, %candidate.offset
%mark.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %candidate.index %mark = load double, ptr addrspace(1) %mark.ptr, align 8 %unmarked = call i1 @recipe.oeq(double %mark, double 0.0)
%score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %candidate.index %score = load double, ptr addrspace(1) %score.ptr, align 8
%none = icmp eq i64 %best, -1 %higher = call i1 @recipe.ogt(double %score, double %best.score) %better = or i1 %none, %higher %take = and i1 %unmarked, %better
%best.next = select i1 %take, i64 %candidate, i64 %best %best.score.next = select i1 %take, double %score, double %best.score %candidate.next = add i64 %candidate, 1 br label %scan.loop
select.mark: %best.offset = mul i64 %best, %length.wide %best.index = add i64 %base, %best.offset %best.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %best.index
store double 1.0, ptr addrspace(1) %best.ptr, align 8 %pick.next = add i64 %pick, 1 br label %select.loop
normalize.entry: br label %max.loop
max.loop: %m = phi i64 [ 0, %normalize.entry ], [ %m.next, %max.step ] %maximum = phi double [ 0.0, %normalize.entry ], [ %maximum.next, %max.step ] %m.first = phi i1 [ true, %normalize.entry ], [ %m.first.next, %max.step ]
%max.more = icmp ult i64 %m, %experts.wide br i1 %max.more, label %max.step, label %sum.entry
max.step: %m.offset = mul i64 %m, %length.wide %m.index = add i64 %base, %m.offset %m.mark.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %m.index %m.mark = load double, ptr addrspace(1) %m.mark.ptr, align 8
%m.marked = call i1 @recipe.oeq(double %m.mark, double 1.0) %m.member = or i1 %m.marked, %every
%m.score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %m.index %m.score = load double, ptr addrspace(1) %m.score.ptr, align 8
%m.higher = call i1 @recipe.ogt(double %m.score, double %maximum) %m.better = or i1 %m.first, %m.higher %m.take = and i1 %m.member, %m.better
%maximum.next = select i1 %m.take, double %m.score, double %maximum %m.first.next = select i1 %m.take, i1 false, i1 %m.first %m.next = add i64 %m, 1 br label %max.loop
sum.entry: br label %sum.loop
sum.loop: %s = phi i64 [ 0, %sum.entry ], [ %s.next, %sum.step ] %total = phi double [ 0.0, %sum.entry ], [ %total.next, %sum.step ] %sum.more = icmp ult i64 %s, %experts.wide br i1 %sum.more, label %sum.step, label %write.entry
sum.step: %s.offset = mul i64 %s, %length.wide %s.index = add i64 %base, %s.offset %s.mark.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %s.index %s.mark = load double, ptr addrspace(1) %s.mark.ptr, align 8
%s.marked = call i1 @recipe.oeq(double %s.mark, double 1.0) %s.member = or i1 %s.marked, %every
%s.score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %s.index %s.score = load double, ptr addrspace(1) %s.score.ptr, align 8
%s.raw = call double @topk_score(double %s.score, double %maximum, i1 %sigmoid) %s.term = select i1 %s.member, double %s.raw, double 0.0
%total.next = call double @recipe.add(double %total, double %s.term) %s.next = add i64 %s, 1 br label %sum.loop
write.entry: %denominator = select i1 %divide, double %total, double 1.0 br label %write.loop
write.loop: %w = phi i64 [ 0, %write.entry ], [ %w.next, %write.step ] %write.more = icmp ult i64 %w, %experts.wide br i1 %write.more, label %write.step, label %exit
write.step: %w.offset = mul i64 %w, %length.wide %w.index = add i64 %base, %w.offset %w.mark.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %w.index %w.mark = load double, ptr addrspace(1) %w.mark.ptr, align 8
%w.marked = call i1 @recipe.oeq(double %w.mark, double 1.0) %w.score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %w.index %w.score = load double, ptr addrspace(1) %w.score.ptr, align 8
%w.raw = call double @topk_score(double %w.score, double %maximum, i1 %sigmoid) %w.probability = call double @recipe.div(double %w.raw, double %denominator)
%w.value = select i1 %w.marked, double %w.probability, double 0.0 store double %w.value, ptr addrspace(1) %w.mark.ptr, align 8 %w.next = add i64 %w, 1 br label %write.loop
exit: ret void }
; The routing adjoint of one position. Each score receives its own slope over
; the divisor times its delta less the kept mean; renormalizing confines that to
; the kept experts, while a plain softmax also reaches the experts it dropped.
define internal void @topk_reverse_body( ptr addrspace(1) %scores, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %experts, i32 %length, i32 %scoring, i32 %renormalize ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%experts.wide = zext i32 %experts to i64 %length.wide = zext i32 %length to i64
%row = udiv i64 %p, %length.wide %position = urem i64 %p, %length.wide %per.row = mul i64 %experts.wide, %length.wide %row.base = mul i64 %row, %per.row %base = add i64 %row.base, %position
%sigmoid = icmp ne i32 %scoring, 0 %renorm = icmp ne i32 %renormalize, 0 %every = xor i1 %renorm, true %plain = xor i1 %sigmoid, true %divide = or i1 %renorm, %plain
br label %max.loop
max.loop: %m = phi i64 [ 0, %entry ], [ %m.next, %max.step ] %maximum = phi double [ 0.0, %entry ], [ %maximum.next, %max.step ] %m.first = phi i1 [ true, %entry ], [ %m.first.next, %max.step ]
%max.more = icmp ult i64 %m, %experts.wide br i1 %max.more, label %max.step, label %sum.entry
max.step: %m.offset = mul i64 %m, %length.wide %m.index = add i64 %base, %m.offset
%m.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %m.index %m.weight = load double, ptr addrspace(1) %m.weight.ptr, align 8
%m.zero = call i1 @recipe.oeq(double %m.weight, double 0.0) %m.marked = xor i1 %m.zero, true %m.member = or i1 %m.marked, %every
%m.score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %m.index %m.score = load double, ptr addrspace(1) %m.score.ptr, align 8
%m.higher = call i1 @recipe.ogt(double %m.score, double %maximum) %m.better = or i1 %m.first, %m.higher %m.take = and i1 %m.member, %m.better
%maximum.next = select i1 %m.take, double %m.score, double %maximum %m.first.next = select i1 %m.take, i1 false, i1 %m.first %m.next = add i64 %m, 1 br label %max.loop
sum.entry: br label %sum.loop
sum.loop: %s = phi i64 [ 0, %sum.entry ], [ %s.next, %sum.step ] %total = phi RECIPE_STATE [ %state.zero, %sum.entry ], [ %total.next, %sum.step ] %inner = phi RECIPE_STATE [ %state.zero, %sum.entry ], [ %inner.next, %sum.step ]
%sum.more = icmp ult i64 %s, %experts.wide br i1 %sum.more, label %sum.step, label %write.entry
sum.step: %s.offset = mul i64 %s, %length.wide %s.index = add i64 %base, %s.offset
%s.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %s.index %s.weight.model = load double, ptr addrspace(1) %s.weight.ptr, align 8 %s.weight = call RECIPE_STATE @recipe.decode(double %s.weight.model)
%s.zero = call i1 @recipe.oeq(double %s.weight.model, double 0.0) %s.marked = xor i1 %s.zero, true %s.member = or i1 %s.marked, %every
%s.score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %s.index %s.score = load double, ptr addrspace(1) %s.score.ptr, align 8
%s.raw.model = call double @topk_score(double %s.score, double %maximum, i1 %sigmoid) %s.raw = call RECIPE_STATE @recipe.decode(double %s.raw.model) %s.term = select i1 %s.member, RECIPE_STATE %s.raw, RECIPE_STATE %state.zero
%total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total, RECIPE_STATE %s.term)
%s.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %s.index %s.delta = load RECIPE_STATE, ptr addrspace(1) %s.delta.ptr, align RECIPE_STATE_ALIGN
%s.product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %s.weight, RECIPE_STATE %s.delta) %inner.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %inner, RECIPE_STATE %s.product) %s.next = add i64 %s, 1 br label %sum.loop
write.entry: %denominator = select i1 %divide, RECIPE_STATE %total, RECIPE_STATE %state.one %subtract = select i1 %divide, RECIPE_STATE %inner, RECIPE_STATE %state.zero br label %write.loop
write.loop: %w = phi i64 [ 0, %write.entry ], [ %w.next, %write.step ] %write.more = icmp ult i64 %w, %experts.wide br i1 %write.more, label %write.step, label %exit
write.step: %w.offset = mul i64 %w, %length.wide %w.index = add i64 %base, %w.offset
%w.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %w.index %w.weight.model = load double, ptr addrspace(1) %w.weight.ptr, align 8 %w.weight = call RECIPE_STATE @recipe.decode(double %w.weight.model)
%w.zero = call i1 @recipe.oeq(double %w.weight.model, double 0.0) %w.marked = xor i1 %w.zero, true
%w.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %w.index %w.delta = load RECIPE_STATE, ptr addrspace(1) %w.delta.ptr, align RECIPE_STATE_ALIGN
%w.score.ptr = getelementptr inbounds double, ptr addrspace(1) %scores, i64 %w.index %w.score = load double, ptr addrspace(1) %w.score.ptr, align 8
%w.raw.model = call double @topk_score(double %w.score, double %maximum, i1 %sigmoid) %w.raw = call RECIPE_STATE @recipe.decode(double %w.raw.model)
%w.rest = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %w.raw) %w.logistic = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %w.raw, RECIPE_STATE %w.rest) %w.slope = select i1 %sigmoid, RECIPE_STATE %w.logistic, RECIPE_STATE %w.raw
%w.coefficient = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %w.slope, RECIPE_STATE %denominator) %w.own = select i1 %w.marked, RECIPE_STATE %w.delta, RECIPE_STATE %state.zero
%w.centered = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %w.own, RECIPE_STATE %subtract) %w.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %w.coefficient, RECIPE_STATE %w.centered)
%w.blocked = and i1 %renorm, %w.zero %w.value = select i1 %w.blocked, RECIPE_STATE %state.zero, RECIPE_STATE %w.term
%w.adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %w.index %w.prior = load RECIPE_STATE, ptr addrspace(1) %w.adjoint.ptr, align RECIPE_STATE_ALIGN
%w.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %w.prior, RECIPE_STATE %w.value) store RECIPE_STATE %w.sum, ptr addrspace(1) %w.adjoint.ptr, align RECIPE_STATE_ALIGN %w.next = add i64 %w, 1 br label %write.loop
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
%bc.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %bc.index %bc.weight = load double, ptr addrspace(1) %bc.ptr, align 8
%bc.zero = call i1 @recipe.oeq(double %bc.weight, double 0.0) %bc.taken = xor i1 %bc.zero, true %bc.step = zext i1 %bc.taken to i32
%bc.step.wide = zext i32 %bc.step to i64 %b.lower.next = add i64 %b.lower, %bc.step.wide %bc.next = add i32 %bc, 1 br label %base.lower
base.done: %b.next = add i32 %b, 1 br label %base.loop
fill.entry: %experts.base = zext i32 %experts to i64 %start = add i64 %experts.base, %base br label %fill.loop
fill.loop: %i = phi i32 [ 0, %fill.entry ], [ %i.next, %advance ] %cursor = phi i32 [ 0, %fill.entry ], [ %cursor.next, %advance ]
%more = icmp ult i32 %i, %pairs br i1 %more, label %step, label %done
step: %i.wide = zext i32 %i to i64 %row = udiv i64 %i.wide, %length.wide %position = urem i64 %i.wide, %length.wide %row.base = mul i64 %row, %per.row %pair.base = add i64 %row.base, %position
%own.index = add i64 %pair.base, %expert.base %own.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %own.index
%own.weight = load double, ptr addrspace(1) %own.ptr, align 8 %own.zero = call i1 @recipe.oeq(double %own.weight, double 0.0) %own.taken = xor i1 %own.zero, true
br i1 %own.taken, label %slot.entry, label %advance
slot.entry: br label %slot.loop
slot.loop: %c = phi i32 [ 0, %slot.entry ], [ %c.next, %slot.step ] %slot = phi i32 [ 0, %slot.entry ], [ %slot.next, %slot.step ]
%c.wide = zext i32 %c to i64 %slot.more = icmp ult i64 %c.wide, %p br i1 %slot.more, label %slot.step, label %write
slot.step: %c.offset = mul i64 %c.wide, %length.wide %c.index = add i64 %pair.base, %c.offset
%c.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %c.index %c.weight = load double, ptr addrspace(1) %c.ptr, align 8
%c.zero = call i1 @recipe.oeq(double %c.weight, double 0.0) %c.taken = xor i1 %c.zero, true %c.step = zext i1 %c.taken to i32
%slot.next = add i32 %slot, %c.step %c.next = add i32 %c, 1 br label %slot.loop
write: %scaled = mul i32 %i, %top %entry.value = add i32 %scaled, %slot %cursor.wide = zext i32 %cursor to i64 %write.index = add i64 %start, %cursor.wide
%write.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %write.index store i32 %entry.value, ptr addrspace(1) %write.ptr, align 4
%cursor.grown = add i32 %cursor, 1 br label %advance
advance: %cursor.next = phi i32 [ %cursor.grown, %write ], [ %cursor, %step ] %i.next = add i32 %i, 1 br label %fill.loop
done: %count.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %p store i32 %cursor, ptr addrspace(1) %count.ptr, align 4 ret void }
; Output element %p of a gate or up projection reads one row of the slice
; belonging to the (slot + 1)-th selected expert of its position.
define internal void @expert_in_forward_body( ptr addrspace(1) %input, ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %decode ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%units = mul i64 %top.wide, %hidden.wide %narrow = mul i64 %units, %length.wide
%row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %unit = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%slot = udiv i64 %unit, %hidden.wide %f = urem i64 %unit, %hidden.wide %slot.i32 = trunc i64 %slot to i32
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
br label %find.loop
find.loop: %c = phi i32 [ 0, %entry ], [ %c.next, %find.step ] %seen = phi i32 [ 0, %entry ], [ %seen.next, %find.step ] %chosen = phi i32 [ -1, %entry ], [ %chosen.next, %find.step ]
%find.more = icmp ult i32 %c, %experts br i1 %find.more, label %find.step, label %sum.entry
find.step: %c.wide = zext i32 %c to i64 %c.offset = mul i64 %c.wide, %length.wide %c.index = add i64 %routing.base, %c.offset
%c.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %c.index %c.weight = load double, ptr addrspace(1) %c.ptr, align 8
%c.zero = call i1 @recipe.oeq(double %c.weight, double 0.0) %c.taken = xor i1 %c.zero, true
%c.match = icmp eq i32 %seen, %slot.i32 %c.unset = icmp eq i32 %chosen, -1 %c.ready = and i1 %c.match, %c.unset %c.pick = and i1 %c.taken, %c.ready
%chosen.next = select i1 %c.pick, i32 %c, i32 %chosen %c.step = zext i1 %c.taken to i32 %seen.next = add i32 %seen, %c.step %c.next = add i32 %c, 1 br label %find.loop
sum.entry: %found = icmp ne i32 %chosen, -1 %expert = select i1 %found, i32 %chosen, i32 0
%plane = mul i64 %hidden.wide, %channels.wide %expert.wide = zext i32 %expert to i64 %slice = mul i64 %expert.wide, %plane %f.row = mul i64 %f, %channels.wide %weight.base = add i64 %slice, %f.row
%input.channels = mul i64 %channels.wide, %length.wide %input.row = mul i64 %row, %input.channels %input.base = add i64 %input.row, %position
br label %sum.loop
sum.loop: %k = phi i32 [ 0, %sum.entry ], [ %k.next, %sum.step ] %total = phi double [ 0.0, %sum.entry ], [ %total.next, %sum.step ]
%sum.more = icmp ult i32 %k, %channels br i1 %sum.more, label %sum.step, label %done
sum.step: %k.wide = zext i32 %k to i64 %weight.index = add i64 %weight.base, %k.wide
%weight = call double @recipe.model.weight(ptr addrspace(1) %weights, i64 %weight.index, i32 %decode)
%input.offset = mul i64 %k.wide, %length.wide %input.index = add i64 %input.base, %input.offset
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %input.index %value = load double, ptr addrspace(1) %input.ptr, align 8
%product = call double @recipe.mul(double %weight, double %value) %total.next = call double @recipe.add(double %total, double %product) %k.next = add i32 %k, 1 br label %sum.loop
done: %output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %total, ptr addrspace(1) %output.ptr, align 8 ret void }
; Input element %p of a gate or up projection collects every selected expert's
; column, in ascending expert order.
define internal void @expert_in_reverse_input_body( ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %channel = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
%units = mul i64 %top.wide, %hidden.wide %delta.narrow = mul i64 %units, %length.wide %delta.row = mul i64 %row, %delta.narrow %delta.base = add i64 %delta.row, %position
%plane = mul i64 %hidden.wide, %channels.wide
br label %expert.loop
expert.loop: %e = phi i32 [ 0, %entry ], [ %e.next, %advance ] %slot = phi i32 [ 0, %entry ], [ %slot.next, %advance ] %total = phi RECIPE_STATE [ %state.zero, %entry ], [ %total.next, %advance ]
%expert.more = icmp ult i32 %e, %experts br i1 %expert.more, label %step, label %done
step: %e.wide = zext i32 %e to i64 %e.offset = mul i64 %e.wide, %length.wide %e.index = add i64 %routing.base, %e.offset
%e.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %e.index %e.weight = load double, ptr addrspace(1) %e.ptr, align 8
%e.zero = call i1 @recipe.oeq(double %e.weight, double 0.0) %e.taken = xor i1 %e.zero, true
br i1 %e.taken, label %hidden.entry, label %advance
hidden.entry: %slice = mul i64 %e.wide, %plane %slot.wide = zext i32 %slot to i64 %slot.offset = mul i64 %slot.wide, %hidden.wide br label %hidden.loop
hidden.loop: %f = phi i32 [ 0, %hidden.entry ], [ %f.next, %hidden.step ] %inner = phi RECIPE_STATE [ %state.zero, %hidden.entry ], [ %inner.next, %hidden.step ]
%hidden.more = icmp ult i32 %f, %hidden br i1 %hidden.more, label %hidden.step, label %hidden.done
hidden.step: %f.wide = zext i32 %f to i64 %unit = add i64 %slot.offset, %f.wide %unit.offset = mul i64 %unit, %length.wide %delta.index = add i64 %delta.base, %unit.offset
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %delta.index %incoming = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%f.row = mul i64 %f.wide, %channels.wide %f.local = add i64 %slice, %f.row %channel.wide = add i64 %channel, 0 %weight.index = add i64 %f.local, %channel.wide
%weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %weight.index %weight.model = load double, ptr addrspace(1) %weight.ptr, align 8 %weight = call RECIPE_STATE @recipe.decode(double %weight.model)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %incoming, RECIPE_STATE %weight) %inner.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %inner, RECIPE_STATE %product) %f.next = add i32 %f, 1 br label %hidden.loop
hidden.done: %grown = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total, RECIPE_STATE %inner) %slot.grown = add i32 %slot, 1 br label %advance
advance: %total.next = phi RECIPE_STATE [ %grown, %hidden.done ], [ %total, %step ] %slot.next = phi i32 [ %slot.grown, %hidden.done ], [ %slot, %step ]
%e.next = add i32 %e, 1 br label %expert.loop
done: %adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %p %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %total) store RECIPE_STATE %sum, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN ret void }
; Weight element %p of a gate or up table sums over the positions routed to its
; own expert, in the bucket order.
define internal void @expert_in_reverse_weight_body( ptr addrspace(1) %input, ptr addrspace(1) %delta, ptr addrspace(1) %context, ptr addrspace(1) %gradient, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %offset ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%plane = mul i64 %hidden.wide, %channels.wide %expert = udiv i64 %p, %plane %local = urem i64 %p, %plane %f = udiv i64 %local, %channels.wide %channel = urem i64 %local, %channels.wide %expert.i32 = trunc i64 %expert to i32
%start = call i32 @moe_bucket_start( ptr addrspace(1) %context, i32 %expert.i32, i32 %experts )
%count.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %expert %count = load i32, ptr addrspace(1) %count.ptr, align 4
%units = mul i64 %top.wide, %hidden.wide %delta.narrow = mul i64 %units, %length.wide %input.narrow = mul i64 %channels.wide, %length.wide
br label %loop
loop: %i = phi i32 [ 0, %entry ], [ %i.next, %step ] %total = phi RECIPE_STATE [ %state.zero, %entry ], [ %total.next, %step ]
%more = icmp ult i32 %i, %count br i1 %more, label %step, label %done
step: %start.wide = zext i32 %start to i64 %i.wide = zext i32 %i to i64 %read.index = add i64 %start.wide, %i.wide %read.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %read.index %packed = load i32, ptr addrspace(1) %read.ptr, align 4
%packed.wide = zext i32 %packed to i64 %pair = udiv i64 %packed.wide, %top.wide %slot = urem i64 %packed.wide, %top.wide %row = udiv i64 %pair, %length.wide %position = urem i64 %pair, %length.wide
%slot.offset = mul i64 %slot, %hidden.wide %unit = add i64 %slot.offset, %f %unit.offset = mul i64 %unit, %length.wide
%delta.row = mul i64 %row, %delta.narrow %delta.local = add i64 %delta.row, %unit.offset %delta.index = add i64 %delta.local, %position
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %delta.index %incoming = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%input.row = mul i64 %row, %input.narrow %channel.offset = mul i64 %channel, %length.wide %input.local = add i64 %input.row, %channel.offset %input.index = add i64 %input.local, %position
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %input.index %value.model = load double, ptr addrspace(1) %input.ptr, align 8 %value = call RECIPE_STATE @recipe.decode(double %value.model)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %incoming, RECIPE_STATE %value) %total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total, RECIPE_STATE %product) %i.next = add i32 %i, 1 br label %loop
done: %offset.wide = zext i32 %offset to i64 %store.index = add i64 %offset.wide, %p %store.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gradient, i64 %store.index store RECIPE_STATE %total, ptr addrspace(1) %store.ptr, align RECIPE_STATE_ALIGN ret void }
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
define internal void @expert_out_forward_body( ptr addrspace(1) %values, ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %output, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %decode ) #1 { entry:
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%narrow = mul i64 %channels.wide, %length.wide %row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %channel = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
%units = mul i64 %top.wide, %hidden.wide %values.narrow = mul i64 %units, %length.wide %values.row = mul i64 %row, %values.narrow %values.base = add i64 %values.row, %position
%plane = mul i64 %channels.wide, %hidden.wide %channel.offset = mul i64 %channel, %hidden.wide
br label %expert.loop
expert.loop: %e = phi i32 [ 0, %entry ], [ %e.next, %advance ] %slot = phi i32 [ 0, %entry ], [ %slot.next, %advance ] %total = phi double [ 0.0, %entry ], [ %total.next, %advance ]
%expert.more = icmp ult i32 %e, %experts br i1 %expert.more, label %step, label %done
step: %e.wide = zext i32 %e to i64 %e.offset = mul i64 %e.wide, %length.wide %e.index = add i64 %routing.base, %e.offset
%e.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %e.index %e.weight = load double, ptr addrspace(1) %e.ptr, align 8
%e.zero = call i1 @recipe.oeq(double %e.weight, double 0.0) %e.taken = xor i1 %e.zero, true
br i1 %e.taken, label %hidden.entry, label %advance
hidden.entry: %slice = mul i64 %e.wide, %plane %weight.base = add i64 %slice, %channel.offset %slot.wide = zext i32 %slot to i64 %slot.offset = mul i64 %slot.wide, %hidden.wide br label %hidden.loop
hidden.loop: %f = phi i32 [ 0, %hidden.entry ], [ %f.next, %hidden.step ] %inner = phi double [ 0.0, %hidden.entry ], [ %inner.next, %hidden.step ]
%hidden.more = icmp ult i32 %f, %hidden br i1 %hidden.more, label %hidden.step, label %hidden.done
hidden.step: %f.wide = zext i32 %f to i64 %weight.index = add i64 %weight.base, %f.wide
%weight = call double @recipe.model.weight(ptr addrspace(1) %weights, i64 %weight.index, i32 %decode)
%unit = add i64 %slot.offset, %f.wide %unit.offset = mul i64 %unit, %length.wide %value.index = add i64 %values.base, %unit.offset
%value.ptr = getelementptr inbounds double, ptr addrspace(1) %values, i64 %value.index %value = load double, ptr addrspace(1) %value.ptr, align 8
%product = call double @recipe.mul(double %weight, double %value) %inner.next = call double @recipe.add(double %inner, double %product) %f.next = add i32 %f, 1 br label %hidden.loop
hidden.done: %scaled = call double @recipe.mul(double %e.weight, double %inner) %grown = call double @recipe.add(double %total, double %scaled) %slot.grown = add i32 %slot, 1 br label %advance
advance: %total.next = phi double [ %grown, %hidden.done ], [ %total, %step ] %slot.next = phi i32 [ %slot.grown, %hidden.done ], [ %slot, %step ]
%e.next = add i32 %e, 1 br label %expert.loop
done: %output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %p store double %total, ptr addrspace(1) %output.ptr, align 8 ret void }
; Hidden element %p of the gated product receives its own expert's column of
; the down table under its routing weight.
define internal void @expert_out_reverse_values_body( ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%units = mul i64 %top.wide, %hidden.wide %narrow = mul i64 %units, %length.wide
%row = udiv i64 %p, %narrow %local = urem i64 %p, %narrow %unit = udiv i64 %local, %length.wide %position = urem i64 %local, %length.wide
%slot = udiv i64 %unit, %hidden.wide %f = urem i64 %unit, %hidden.wide %slot.i32 = trunc i64 %slot to i32
%per.row = mul i64 %experts.wide, %length.wide %routing.row = mul i64 %row, %per.row %routing.base = add i64 %routing.row, %position
br label %find.loop
find.loop: %c = phi i32 [ 0, %entry ], [ %c.next, %find.step ] %seen = phi i32 [ 0, %entry ], [ %seen.next, %find.step ] %chosen = phi i32 [ -1, %entry ], [ %chosen.next, %find.step ]
%find.more = icmp ult i32 %c, %experts br i1 %find.more, label %find.step, label %sum.entry
find.step: %c.wide = zext i32 %c to i64 %c.offset = mul i64 %c.wide, %length.wide %c.index = add i64 %routing.base, %c.offset
%c.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %c.index %c.weight = load double, ptr addrspace(1) %c.ptr, align 8
%c.zero = call i1 @recipe.oeq(double %c.weight, double 0.0) %c.taken = xor i1 %c.zero, true
%c.match = icmp eq i32 %seen, %slot.i32 %c.unset = icmp eq i32 %chosen, -1 %c.ready = and i1 %c.match, %c.unset %c.pick = and i1 %c.taken, %c.ready
%chosen.next = select i1 %c.pick, i32 %c, i32 %chosen %c.step = zext i1 %c.taken to i32 %seen.next = add i32 %seen, %c.step %c.next = add i32 %c, 1 br label %find.loop
sum.entry: %found = icmp ne i32 %chosen, -1 %expert = select i1 %found, i32 %chosen, i32 0
%expert.wide = zext i32 %expert to i64 %expert.offset = mul i64 %expert.wide, %length.wide %expert.index = add i64 %routing.base, %expert.offset
%expert.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %expert.index %routed.model = load double, ptr addrspace(1) %expert.ptr, align 8 %routed = call RECIPE_STATE @recipe.decode(double %routed.model)
%plane = mul i64 %channels.wide, %hidden.wide %slice = mul i64 %expert.wide, %plane
%delta.narrow = mul i64 %channels.wide, %length.wide %delta.row = mul i64 %row, %delta.narrow %delta.base = add i64 %delta.row, %position
br label %sum.loop
sum.loop: %o = phi i32 [ 0, %sum.entry ], [ %o.next, %sum.step ] %total = phi RECIPE_STATE [ %state.zero, %sum.entry ], [ %total.next, %sum.step ]
%sum.more = icmp ult i32 %o, %channels br i1 %sum.more, label %sum.step, label %done
sum.step: %o.wide = zext i32 %o to i64 %o.offset = mul i64 %o.wide, %length.wide %delta.index = add i64 %delta.base, %o.offset
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %delta.index %incoming = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%o.row = mul i64 %o.wide, %hidden.wide %o.local = add i64 %slice, %o.row %weight.index = add i64 %o.local, %f
%weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %weight.index %weight.model = load double, ptr addrspace(1) %weight.ptr, align 8 %weight = call RECIPE_STATE @recipe.decode(double %weight.model)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %incoming, RECIPE_STATE %weight) %total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total, RECIPE_STATE %product) %o.next = add i32 %o, 1 br label %sum.loop
done: %scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %routed, RECIPE_STATE %total) %result = select i1 %found, RECIPE_STATE %scaled, RECIPE_STATE %state.zero
%adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %p %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %result) store RECIPE_STATE %sum, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN ret void }
; Weight element %p of the down table sums over the positions routed to its own
; expert, in the bucket order.
define internal void @expert_out_reverse_weight_body( ptr addrspace(1) %values, ptr addrspace(1) %routing, ptr addrspace(1) %delta, ptr addrspace(1) %context, ptr addrspace(1) %gradient, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top, i32 %offset ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%channels.wide = zext i32 %channels to i64 %length.wide = zext i32 %length to i64 %hidden.wide = zext i32 %hidden to i64 %experts.wide = zext i32 %experts to i64 %top.wide = zext i32 %top to i64
%plane = mul i64 %channels.wide, %hidden.wide %expert = udiv i64 %p, %plane %local = urem i64 %p, %plane %channel = udiv i64 %local, %hidden.wide %f = urem i64 %local, %hidden.wide %expert.i32 = trunc i64 %expert to i32
%start = call i32 @moe_bucket_start( ptr addrspace(1) %context, i32 %expert.i32, i32 %experts )
%count.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i64 %expert %count = load i32, ptr addrspace(1) %count.ptr, align 4
%units = mul i64 %top.wide, %hidden.wide %values.narrow = mul i64 %units, %length.wide %delta.narrow = mul i64 %channels.wide, %length.wide %per.row = mul i64 %experts.wide, %length.wide
%expert.offset = mul i64 %expert, %length.wide %channel.offset = mul i64 %channel, %length.wide
br label %loop
loop: %i = phi i32 [ 0, %entry ], [ %i.next, %step ] %total = phi RECIPE_STATE [ %state.zero, %entry ], [ %total.next, %step ]
%more = icmp ult i32 %i, %count br i1 %more, label %step, label %done
step: %read.index = add i32 %start, %i %read.ptr = getelementptr inbounds i32, ptr addrspace(1) %context, i32 %read.index %packed = load i32, ptr addrspace(1) %read.ptr, align 4
%pair = udiv i32 %packed, %top %slot = urem i32 %packed, %top %row = udiv i32 %pair, %length %position = urem i32 %pair, %length
%row.wide = zext i32 %row to i64 %position.wide = zext i32 %position to i64 %routing.row = mul i64 %row.wide, %per.row %routing.local = add i64 %routing.row, %expert.offset %routing.index = add i64 %routing.local, %position.wide
%routing.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %routing.index %routed.model = load double, ptr addrspace(1) %routing.ptr, align 8 %routed = call RECIPE_STATE @recipe.decode(double %routed.model)
%delta.row = mul i64 %row.wide, %delta.narrow %delta.channel.offset = mul i64 %channel, %length.wide %delta.local = add i64 %delta.row, %delta.channel.offset %delta.index = add i64 %delta.local, %position.wide
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %delta.index %incoming = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%slot.wide = zext i32 %slot to i64 %slot.offset = mul i64 %slot.wide, %hidden.wide %unit = add i64 %slot.offset, %f %unit.offset = mul i64 %unit, %length.wide
%values.row = mul i64 %row.wide, %values.narrow %values.local = add i64 %values.row, %unit.offset %values.index = add i64 %values.local, %position.wide
%values.ptr = getelementptr inbounds double, ptr addrspace(1) %values, i64 %values.index %value.model = load double, ptr addrspace(1) %values.ptr, align 8 %value = call RECIPE_STATE @recipe.decode(double %value.model)
%weighted = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %routed, RECIPE_STATE %incoming) %product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %weighted, RECIPE_STATE %value)
%total.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total, RECIPE_STATE %product) %i.next = add i32 %i, 1 br label %loop
done: %offset.wide = zext i32 %offset to i64 %store.index = add i64 %offset.wide, %p %store.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gradient, i64 %store.index store RECIPE_STATE %total, ptr addrspace(1) %store.ptr, align RECIPE_STATE_ALIGN ret void }
; Position %p sends each selected expert's down projection back to its own
; routing weight, in ascending expert order.
define internal void @expert_out_reverse_routing_body( ptr addrspace(1) %values, ptr addrspace(1) %routing, ptr addrspace(1) %weights, ptr addrspace(1) %delta, ptr addrspace(1) %adjoint, i64 %p, i32 %channels, i32 %length, i32 %hidden, i32 %experts, i32 %top ) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
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
%e.ptr = getelementptr inbounds double, ptr addrspace(1) %routing, i64 %e.index %e.weight = load double, ptr addrspace(1) %e.ptr, align 8
%e.zero = call i1 @recipe.oeq(double %e.weight, double 0.0) %e.taken = xor i1 %e.zero, true
br i1 %e.taken, label %channel.entry, label %advance
channel.entry: %slice = mul i64 %e.wide, %plane %slot.wide = zext i32 %slot to i64 %slot.offset = mul i64 %slot.wide, %hidden.wide br label %channel.loop
channel.loop: %o = phi i32 [ 0, %channel.entry ], [ %o.next, %channel.done ] %outer = phi RECIPE_STATE [ %state.zero, %channel.entry ], [ %outer.next, %channel.done ]
%channel.more = icmp ult i32 %o, %channels br i1 %channel.more, label %channel.step, label %write
channel.step: %o.wide = zext i32 %o to i64 %o.offset = mul i64 %o.wide, %length.wide %delta.index = add i64 %delta.base, %o.offset
%delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %delta.index %incoming = load RECIPE_STATE, ptr addrspace(1) %delta.ptr, align RECIPE_STATE_ALIGN
%o.row = mul i64 %o.wide, %hidden.wide %weight.base = add i64 %slice, %o.row br label %hidden.loop
hidden.loop: %f = phi i32 [ 0, %channel.step ], [ %f.next, %hidden.step ] %inner = phi RECIPE_STATE [ %state.zero, %channel.step ], [ %inner.next, %hidden.step ]
%hidden.more = icmp ult i32 %f, %hidden br i1 %hidden.more, label %hidden.step, label %channel.done
hidden.step: %f.wide = zext i32 %f to i64 %weight.index = add i64 %weight.base, %f.wide %weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %weight.index
%weight.model = load double, ptr addrspace(1) %weight.ptr, align 8 %weight = call RECIPE_STATE @recipe.decode(double %weight.model)
%unit = add i64 %slot.offset, %f.wide %unit.offset = mul i64 %unit, %length.wide %value.index = add i64 %values.base, %unit.offset
%value.ptr = getelementptr inbounds double, ptr addrspace(1) %values, i64 %value.index %value.model = load double, ptr addrspace(1) %value.ptr, align 8 %value = call RECIPE_STATE @recipe.decode(double %value.model)
%product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %weight, RECIPE_STATE %value) %inner.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %inner, RECIPE_STATE %product) %f.next = add i32 %f, 1 br label %hidden.loop
channel.done: %term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %incoming, RECIPE_STATE %inner) %outer.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %outer, RECIPE_STATE %term) %o.next = add i32 %o, 1 br label %channel.loop
write: %adjoint.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %adjoint, i64 %e.index %prior = load RECIPE_STATE, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %prior, RECIPE_STATE %outer) store RECIPE_STATE %sum, ptr addrspace(1) %adjoint.ptr, align RECIPE_STATE_ALIGN
%slot.grown = add i32 %slot, 1 br label %advance
advance: %slot.next = phi i32 [ %slot.grown, %write ], [ %slot, %step ] %e.next = add i32 %e, 1 br label %expert.loop
done: ret void }
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
define internal RECIPE_STATE @attention_tile_dot_state(i32 %left, i32 %right, i32 %width, i32 %left.base, i32 %right.base) #1 { entry:
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
%left.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %left.index
%left.wide = load RECIPE_STATE, ptr addrspace(3) %left.ptr, align RECIPE_STATE_ALIGN
%right.row = mul i32 %right, %width
%right.local = add i32 %right.row, %channel
%right.index = add i32 %right.base, %right.local
%right.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %right.index
%right.wide = load RECIPE_STATE, ptr addrspace(3) %right.ptr, align RECIPE_STATE_ALIGN
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
define internal void @attention_step_score_store(ptr addrspace(1) %context, i64 %index, RECIPE_STATE %value) #1 { entry:
%ptr = getelementptr RECIPE_STATE, ptr addrspace(1) %context, i64 %index
store RECIPE_STATE %value, ptr addrspace(1) %ptr, align RECIPE_STATE_ALIGN
ret void
}
define internal RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %index) #1 { entry:
%ptr = getelementptr RECIPE_STATE, ptr addrspace(1) %context, i64 %index
%value = load RECIPE_STATE, ptr addrspace(1) %ptr, align RECIPE_STATE_ALIGN
ret RECIPE_STATE %value
}
; Whole-grid inference attention for one query. The ordinary context's two
; statistics planes temporarily hold one packed state RECIPE_STATE per score; the
; dedicated K/V context keeps the settled history in the cache type.
define internal RECIPE_STATE @attention_step_key_dot(ptr addrspace(3) %query, ptr addrspace(1) %kv.context, i32 %key, i32 %kv.head, i32 %width, i32 %length) #1 {
entry:
%kv.channel.base = mul i32 %kv.head, %width
br label %loop
loop:
%channel = phi i32 [ 0, %entry ], [ %channel.next, %step ]
%sum.0 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.0.next, %step ]
%sum.1 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.1.next, %step ]
%sum.2 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.2.next, %step ]
%sum.3 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.3.next, %step ]
%sum.4 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.4.next, %step ]
%sum.5 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.5.next, %step ]
%sum.6 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.6.next, %step ]
%sum.7 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.7.next, %step ]
%sum.8 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.8.next, %step ]
%sum.9 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.9.next, %step ]
%sum.10 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.10.next, %step ]
%sum.11 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.11.next, %step ]
%sum.12 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.12.next, %step ]
%sum.13 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.13.next, %step ]
%sum.14 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.14.next, %step ]
%sum.15 = phi RECIPE_STATE [ 0x0000000000000000, %entry ], [ %sum.15.next, %step ]
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
%q.0.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.0
%q.1.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.1
%q.2.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.2
%q.3.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.3
%q.4.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.4
%q.5.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.5
%q.6.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.6
%q.7.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.7
%q.8.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.8
%q.9.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.9
%q.10.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.10
%q.11.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.11
%q.12.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.12
%q.13.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.13
%q.14.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.14
%q.15.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %query, i32 %safe.15
%q.0.raw = load RECIPE_STATE, ptr addrspace(3) %q.0.ptr, align RECIPE_STATE_ALIGN
%q.1.raw = load RECIPE_STATE, ptr addrspace(3) %q.1.ptr, align RECIPE_STATE_ALIGN
%q.2.raw = load RECIPE_STATE, ptr addrspace(3) %q.2.ptr, align RECIPE_STATE_ALIGN
%q.3.raw = load RECIPE_STATE, ptr addrspace(3) %q.3.ptr, align RECIPE_STATE_ALIGN
%q.4.raw = load RECIPE_STATE, ptr addrspace(3) %q.4.ptr, align RECIPE_STATE_ALIGN
%q.5.raw = load RECIPE_STATE, ptr addrspace(3) %q.5.ptr, align RECIPE_STATE_ALIGN
%q.6.raw = load RECIPE_STATE, ptr addrspace(3) %q.6.ptr, align RECIPE_STATE_ALIGN
%q.7.raw = load RECIPE_STATE, ptr addrspace(3) %q.7.ptr, align RECIPE_STATE_ALIGN
%q.8.raw = load RECIPE_STATE, ptr addrspace(3) %q.8.ptr, align RECIPE_STATE_ALIGN
%q.9.raw = load RECIPE_STATE, ptr addrspace(3) %q.9.ptr, align RECIPE_STATE_ALIGN
%q.10.raw = load RECIPE_STATE, ptr addrspace(3) %q.10.ptr, align RECIPE_STATE_ALIGN
%q.11.raw = load RECIPE_STATE, ptr addrspace(3) %q.11.ptr, align RECIPE_STATE_ALIGN
%q.12.raw = load RECIPE_STATE, ptr addrspace(3) %q.12.ptr, align RECIPE_STATE_ALIGN
%q.13.raw = load RECIPE_STATE, ptr addrspace(3) %q.13.ptr, align RECIPE_STATE_ALIGN
%q.14.raw = load RECIPE_STATE, ptr addrspace(3) %q.14.ptr, align RECIPE_STATE_ALIGN
%q.15.raw = load RECIPE_STATE, ptr addrspace(3) %q.15.ptr, align RECIPE_STATE_ALIGN
%q.0 = select i1 %active.0, RECIPE_STATE %q.0.raw, RECIPE_STATE 0x0000000000000000
%q.1 = select i1 %active.1, RECIPE_STATE %q.1.raw, RECIPE_STATE 0x0000000000000000
%q.2 = select i1 %active.2, RECIPE_STATE %q.2.raw, RECIPE_STATE 0x0000000000000000
%q.3 = select i1 %active.3, RECIPE_STATE %q.3.raw, RECIPE_STATE 0x0000000000000000
%q.4 = select i1 %active.4, RECIPE_STATE %q.4.raw, RECIPE_STATE 0x0000000000000000
%q.5 = select i1 %active.5, RECIPE_STATE %q.5.raw, RECIPE_STATE 0x0000000000000000
%q.6 = select i1 %active.6, RECIPE_STATE %q.6.raw, RECIPE_STATE 0x0000000000000000
%q.7 = select i1 %active.7, RECIPE_STATE %q.7.raw, RECIPE_STATE 0x0000000000000000
%q.8 = select i1 %active.8, RECIPE_STATE %q.8.raw, RECIPE_STATE 0x0000000000000000
%q.9 = select i1 %active.9, RECIPE_STATE %q.9.raw, RECIPE_STATE 0x0000000000000000
%q.10 = select i1 %active.10, RECIPE_STATE %q.10.raw, RECIPE_STATE 0x0000000000000000
%q.11 = select i1 %active.11, RECIPE_STATE %q.11.raw, RECIPE_STATE 0x0000000000000000
%q.12 = select i1 %active.12, RECIPE_STATE %q.12.raw, RECIPE_STATE 0x0000000000000000
%q.13 = select i1 %active.13, RECIPE_STATE %q.13.raw, RECIPE_STATE 0x0000000000000000
%q.14 = select i1 %active.14, RECIPE_STATE %q.14.raw, RECIPE_STATE 0x0000000000000000
%q.15 = select i1 %active.15, RECIPE_STATE %q.15.raw, RECIPE_STATE 0x0000000000000000
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
%key.0.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.0.wide
%key.1.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.1.wide
%key.2.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.2.wide
%key.3.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.3.wide
%key.4.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.4.wide
%key.5.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.5.wide
%key.6.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.6.wide
%key.7.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.7.wide
%key.8.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.8.wide
%key.9.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.9.wide
%key.10.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.10.wide
%key.11.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.11.wide
%key.12.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.12.wide
%key.13.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.13.wide
%key.14.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.14.wide
%key.15.ptr = getelementptr RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.15.wide
%key.0.load = load RECIPE_KV, ptr addrspace(1) %key.0.ptr, align RECIPE_KV_ALIGN
%key.1.load = load RECIPE_KV, ptr addrspace(1) %key.1.ptr, align RECIPE_KV_ALIGN
%key.2.load = load RECIPE_KV, ptr addrspace(1) %key.2.ptr, align RECIPE_KV_ALIGN
%key.3.load = load RECIPE_KV, ptr addrspace(1) %key.3.ptr, align RECIPE_KV_ALIGN
%key.4.load = load RECIPE_KV, ptr addrspace(1) %key.4.ptr, align RECIPE_KV_ALIGN
%key.5.load = load RECIPE_KV, ptr addrspace(1) %key.5.ptr, align RECIPE_KV_ALIGN
%key.6.load = load RECIPE_KV, ptr addrspace(1) %key.6.ptr, align RECIPE_KV_ALIGN
%key.7.load = load RECIPE_KV, ptr addrspace(1) %key.7.ptr, align RECIPE_KV_ALIGN
%key.8.load = load RECIPE_KV, ptr addrspace(1) %key.8.ptr, align RECIPE_KV_ALIGN
%key.9.load = load RECIPE_KV, ptr addrspace(1) %key.9.ptr, align RECIPE_KV_ALIGN
%key.10.load = load RECIPE_KV, ptr addrspace(1) %key.10.ptr, align RECIPE_KV_ALIGN
%key.11.load = load RECIPE_KV, ptr addrspace(1) %key.11.ptr, align RECIPE_KV_ALIGN
%key.12.load = load RECIPE_KV, ptr addrspace(1) %key.12.ptr, align RECIPE_KV_ALIGN
%key.13.load = load RECIPE_KV, ptr addrspace(1) %key.13.ptr, align RECIPE_KV_ALIGN
%key.14.load = load RECIPE_KV, ptr addrspace(1) %key.14.ptr, align RECIPE_KV_ALIGN
%key.15.load = load RECIPE_KV, ptr addrspace(1) %key.15.ptr, align RECIPE_KV_ALIGN
%key.0.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.0.load)
%key.1.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.1.load)
%key.2.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.2.load)
%key.3.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.3.load)
%key.4.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.4.load)
%key.5.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.5.load)
%key.6.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.6.load)
%key.7.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.7.load)
%key.8.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.8.load)
%key.9.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.9.load)
%key.10.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.10.load)
%key.11.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.11.load)
%key.12.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.12.load)
%key.13.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.13.load)
%key.14.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.14.load)
%key.15.raw = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %key.15.load)
%key.0 = select i1 %active.0, RECIPE_STATE %key.0.raw, RECIPE_STATE 0x0000000000000000
%key.1 = select i1 %active.1, RECIPE_STATE %key.1.raw, RECIPE_STATE 0x0000000000000000
%key.2 = select i1 %active.2, RECIPE_STATE %key.2.raw, RECIPE_STATE 0x0000000000000000
%key.3 = select i1 %active.3, RECIPE_STATE %key.3.raw, RECIPE_STATE 0x0000000000000000
%key.4 = select i1 %active.4, RECIPE_STATE %key.4.raw, RECIPE_STATE 0x0000000000000000
%key.5 = select i1 %active.5, RECIPE_STATE %key.5.raw, RECIPE_STATE 0x0000000000000000
%key.6 = select i1 %active.6, RECIPE_STATE %key.6.raw, RECIPE_STATE 0x0000000000000000
%key.7 = select i1 %active.7, RECIPE_STATE %key.7.raw, RECIPE_STATE 0x0000000000000000
%key.8 = select i1 %active.8, RECIPE_STATE %key.8.raw, RECIPE_STATE 0x0000000000000000
%key.9 = select i1 %active.9, RECIPE_STATE %key.9.raw, RECIPE_STATE 0x0000000000000000
%key.10 = select i1 %active.10, RECIPE_STATE %key.10.raw, RECIPE_STATE 0x0000000000000000
%key.11 = select i1 %active.11, RECIPE_STATE %key.11.raw, RECIPE_STATE 0x0000000000000000
%key.12 = select i1 %active.12, RECIPE_STATE %key.12.raw, RECIPE_STATE 0x0000000000000000
%key.13 = select i1 %active.13, RECIPE_STATE %key.13.raw, RECIPE_STATE 0x0000000000000000
%key.14 = select i1 %active.14, RECIPE_STATE %key.14.raw, RECIPE_STATE 0x0000000000000000
%key.15 = select i1 %active.15, RECIPE_STATE %key.15.raw, RECIPE_STATE 0x0000000000000000
%sum.0.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.0, RECIPE_STATE %q.0, RECIPE_STATE %key.0)
%sum.1.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.1, RECIPE_STATE %q.1, RECIPE_STATE %key.1)
%sum.2.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.2, RECIPE_STATE %q.2, RECIPE_STATE %key.2)
%sum.3.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.3, RECIPE_STATE %q.3, RECIPE_STATE %key.3)
%sum.4.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.4, RECIPE_STATE %q.4, RECIPE_STATE %key.4)
%sum.5.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.5, RECIPE_STATE %q.5, RECIPE_STATE %key.5)
%sum.6.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.6, RECIPE_STATE %q.6, RECIPE_STATE %key.6)
%sum.7.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.7, RECIPE_STATE %q.7, RECIPE_STATE %key.7)
%sum.8.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.8, RECIPE_STATE %q.8, RECIPE_STATE %key.8)
%sum.9.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.9, RECIPE_STATE %q.9, RECIPE_STATE %key.9)
%sum.10.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.10, RECIPE_STATE %q.10, RECIPE_STATE %key.10)
%sum.11.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.11, RECIPE_STATE %q.11, RECIPE_STATE %key.11)
%sum.12.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.12, RECIPE_STATE %q.12, RECIPE_STATE %key.12)
%sum.13.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.13, RECIPE_STATE %q.13, RECIPE_STATE %key.13)
%sum.14.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.14, RECIPE_STATE %q.14, RECIPE_STATE %key.14)
%sum.15.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %sum.15, RECIPE_STATE %q.15, RECIPE_STATE %key.15)
%channel.next = add i32 %channel, 16
br label %loop
done:
%sum.89 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.8, RECIPE_STATE %sum.9)
%sum.1011 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.10, RECIPE_STATE %sum.11)
%sum.1213 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.12, RECIPE_STATE %sum.13)
%sum.1415 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.14, RECIPE_STATE %sum.15)
%sum.89.1011 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.89, RECIPE_STATE %sum.1011)
%sum.1213.1415 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.1213, RECIPE_STATE %sum.1415)
%sum.8to15 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.89.1011, RECIPE_STATE %sum.1213.1415)
%sum.01 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.0, RECIPE_STATE %sum.1)
%sum.23 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.2, RECIPE_STATE %sum.3)
%sum.45 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.4, RECIPE_STATE %sum.5)
%sum.67 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.6, RECIPE_STATE %sum.7)
%sum.0123 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.01, RECIPE_STATE %sum.23)
%sum.4567 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.45, RECIPE_STATE %sum.67)
%sum.0to7 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.0123, RECIPE_STATE %sum.4567)
%sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum.0to7, RECIPE_STATE %sum.8to15)
ret RECIPE_STATE %sum
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
i32 %from, i32 %heads, i32 %channels, i32 %position, i32 %kv.heads, i32 %threads, i32 %buffer.length, i32 %buffer.origin, i32 %select.block, i32 %index.width) #3 {
entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%global = mul i32 %group, %block
%global.id = add i32 %global, %lid
%length = udiv i32 %from, %channels
%width = udiv i32 %channels, %heads
%kv.group = udiv i32 %heads, %kv.heads
%kv.channels = mul i32 %kv.heads, %width
%kv.plane = mul i32 %kv.channels, %length
%reached = add i32 %position, 1
; With a selection block, the indexer's flags keep some blocks of keys for
; this query; they sit past the statistics and the block representatives.
%select.on = icmp ne i32 %select.block, 0
%select.divisor = select i1 %select.on, i32 %select.block, i32 1
%select.blocks.over = add i32 %length, %select.divisor
%select.blocks.less = sub i32 %select.blocks.over, 1
%select.blocks = udiv i32 %select.blocks.less, %select.divisor
%select.statistics = mul i32 %heads, %length
%select.representatives = mul i32 %select.statistics, 2
%select.representative.total = mul i32 %select.blocks, %index.width
%select.base = add i32 %select.representatives, %select.representative.total
%select.base.wide = zext i32 %select.base to i64
%wave.width = call i32 @recipe.wavefront.width()
%wave = udiv i32 %lid, %wave.width
%lane = urem i32 %lid, %wave.width
%waves = udiv i32 %block, %wave.width
%global.wave = udiv i32 %global.id, %wave.width
%global.waves = udiv i32 %threads, %wave.width
%width.RECIPE_STATE = call RECIPE_STATE @recipe.state.from.u32(i32 %width)
%scale = call RECIPE_STATE @recipe.state.sqrt(RECIPE_STATE %width.RECIPE_STATE)
; The running maximum starts below every score: minus infinity from the
; state's own division, so no literal names a type.
%floor.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%floor.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%floor.minus = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %floor.one)
%score.floor = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %floor.minus, RECIPE_STATE %floor.zero)
%wave.half.start = lshr i32 %wave.width, 1
%maximum.global.index = add i32 %waves, 0
%denominator.global.index = add i32 %waves, 1
%tile.base = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 0
%tile.RECIPE_STATE = bitcast ptr addrspace(3) %tile.base to ptr addrspace(3)
%maximum.global.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %tile.RECIPE_STATE, i32 %maximum.global.index
%denominator.global.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %tile.RECIPE_STATE, i32 %denominator.global.index
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
%cache.key.source.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %cache.key.source.phys
%cache.key.value = load double, ptr addrspace(1) %cache.key.source.ptr, align 8
%cache.value.base = add i32 %from, %kv.plane
%cache.value.source.index = add i32 %cache.value.base, %cache.position.index
%cache.value.source.wide = zext i32 %cache.value.source.index to i64
%cache.value.source.phys = call i64 @recipe.window.index(i64 %cache.value.source.wide, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%cache.value.source.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %cache.value.source.phys
%cache.value.value = load double, ptr addrspace(1) %cache.value.source.ptr, align 8
%cache.key.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %cache.position.wide
%cache.key.kv = call RECIPE_KV @recipe.kv.encode(double %cache.key.value)
store RECIPE_KV %cache.key.kv, ptr addrspace(1) %cache.key.ptr, align RECIPE_KV_ALIGN
%cache.value.index = add i32 %kv.plane, %cache.position.index
%cache.value.wide = zext i32 %cache.value.index to i64
%cache.value.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %cache.value.wide
%cache.value.kv = call RECIPE_KV @recipe.kv.encode(double %cache.value.value)
store RECIPE_KV %cache.value.kv, ptr addrspace(1) %cache.value.ptr, align RECIPE_KV_ALIGN
%cache.channel.next = add i32 %cache.channel, %threads
br label %cache.loop
cache.done:
call void @grid_barrier(i32 %threads)
; Each workgroup scores one head per round; every workgroup runs the same
; number of rounds, so a grid with fewer workgroups than heads reaches every
; head and the grid barriers between phases stay uniform.
%groups = udiv i32 %threads, %block
%rounds.span = add i32 %heads, %groups
%rounds.full = sub i32 %rounds.span, 1
%rounds = udiv i32 %rounds.full, %groups
br label %head.loop
head.loop:
%round = phi i32 [ 0, %cache.done ], [ %round.next, %probability.done ]
%head.more = icmp ult i32 %round, %rounds
br i1 %head.more, label %head.start, label %head.done
head.start:
%round.offset = mul i32 %round, %groups
%head = add i32 %group, %round.offset
br label %score.tile.loop
score.tile.loop:
%score.tile.base = phi i32 [ 0, %head.start ], [ %score.tile.next, %score.tile.advance ]
%score.group.active = icmp ult i32 %head, %heads
%score.tile.limit = icmp ult i32 %score.tile.base, %reached
%score.tile.more = and i1 %score.group.active, %score.tile.limit
br i1 %score.tile.more, label %score.query.copy.loop, label %score.done
score.query.copy.loop:
%score.query.channel = phi i32 [ %lid, %score.tile.loop ], [ %score.query.channel.next, %score.query.copy.step ]
%score.query.channel.more = icmp ult i32 %score.query.channel, %width
br i1 %score.query.channel.more, label %score.query.copy.step, label %score.query.copy.done
score.query.copy.step:
%score.head.channel = mul i32 %head, %width
%score.query.global.channel = add i32 %score.head.channel, %score.query.channel
%score.query.channel.base = mul i32 %score.query.global.channel, %length
%score.query.index = add i32 %score.query.channel.base, %position
%score.query.wide = zext i32 %score.query.index to i64
%score.query.phys = call i64 @recipe.window.index(i64 %score.query.wide, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%score.query.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %score.query.phys
%score.query.value = load double, ptr addrspace(1) %score.query.ptr, align 8
%score.query.RECIPE_STATE = call RECIPE_STATE @recipe.state.from.model(double %score.query.value)
%score.query.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %tile.RECIPE_STATE, i32 %score.query.channel
store RECIPE_STATE %score.query.RECIPE_STATE, ptr addrspace(3) %score.query.shared.ptr, align RECIPE_STATE_ALIGN
%score.query.channel.next = add i32 %score.query.channel, %block
br label %score.query.copy.loop
score.query.copy.done:
call void @recipe.local.barrier()
%score.key = add i32 %score.tile.base, %lid
%score.key.more = icmp ult i32 %score.key, %reached
br i1 %score.key.more, label %score.key.compute, label %score.key.done
score.key.compute:
%score.kv.head = udiv i32 %head, %kv.group
%score.scaled.raw = call RECIPE_STATE @attention_step_key_dot(ptr addrspace(3) %tile.RECIPE_STATE, ptr addrspace(1) %kv.context, i32 %score.key, i32 %score.kv.head, i32 %width, i32 %length)
%score.scaled = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %score.scaled.raw, RECIPE_STATE %scale)
br i1 %select.on, label %score.select, label %score.selected
score.select:
%score.kept.flag = call i1 @attention_selected(ptr addrspace(1) %context, i64 %select.base.wide, i32 %select.blocks, i32 %select.divisor, i32 %position, i32 %score.key)
br label %score.selected
score.selected:
%score.kept = phi i1 [ true, %score.key.compute ], [ %score.kept.flag, %score.select ]
%score.masked = select i1 %score.kept, RECIPE_STATE %score.scaled, RECIPE_STATE %score.floor
%score.row = mul i32 %head, %length
%score.slot = add i32 %score.row, %score.key
%score.slot.wide = zext i32 %score.slot to i64
call void @attention_step_score_store(ptr addrspace(1) %context, i64 %score.slot.wide, RECIPE_STATE %score.masked)
br label %score.key.done
score.key.done:
call void @recipe.local.barrier()
br label %score.tile.advance
score.tile.advance:
%score.tile.next = add i32 %score.tile.base, %block
br label %score.tile.loop
score.done:
call void @grid_barrier(i32 %threads)
%active = icmp ult i32 %head, %heads
%active.limit = select i1 %active, i32 %reached, i32 0
br label %maximum.loop
maximum.loop:
%maximum.key = phi i32 [ %lid, %score.done ], [ %maximum.key.next, %maximum.step ]
%maximum.value = phi RECIPE_STATE [ %score.floor, %score.done ], [ %maximum.next, %maximum.step ]
%maximum.more = icmp ult i32 %maximum.key, %active.limit
br i1 %maximum.more, label %maximum.step, label %maximum.wave.loop
maximum.step:
%maximum.row = mul i32 %head, %length
%maximum.slot = add i32 %maximum.row, %maximum.key
%maximum.slot.wide = zext i32 %maximum.slot to i64
%maximum.score = call RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %maximum.slot.wide)
%maximum.larger = call i1 @recipe.state.ogt(RECIPE_STATE %maximum.score, RECIPE_STATE %maximum.value)
%maximum.next = select i1 %maximum.larger, RECIPE_STATE %maximum.score, RECIPE_STATE %maximum.value
%maximum.key.next = add i32 %maximum.key, %block
br label %maximum.loop
maximum.wave.loop:
%maximum.offset = phi i32 [ %wave.half.start, %maximum.loop ], [ %maximum.offset.next, %maximum.wave.step ]
%maximum.wave.value = phi RECIPE_STATE [ %maximum.value, %maximum.loop ], [ %maximum.wave.value.next, %maximum.wave.step ]
%maximum.wave.more = icmp ne i32 %maximum.offset, 0
br i1 %maximum.wave.more, label %maximum.wave.step, label %maximum.wave.done
maximum.wave.step:
%maximum.partner.lane = xor i32 %lane, %maximum.offset
%maximum.partner.index = mul i32 %maximum.partner.lane, 4
%maximum.partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %maximum.wave.value, i32 %maximum.partner.index)
%maximum.partner.larger = call i1 @recipe.state.ogt(RECIPE_STATE %maximum.partner, RECIPE_STATE %maximum.wave.value)
%maximum.wave.value.next = select i1 %maximum.partner.larger, RECIPE_STATE %maximum.partner, RECIPE_STATE %maximum.wave.value
%maximum.offset.next = lshr i32 %maximum.offset, 1
br label %maximum.wave.loop
maximum.wave.done:
%maximum.owner = icmp eq i32 %lane, 0
br i1 %maximum.owner, label %maximum.wave.store, label %maximum.wave.skip
maximum.wave.store:
%maximum.wave.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %tile.RECIPE_STATE, i32 %wave
store RECIPE_STATE %maximum.wave.value, ptr addrspace(3) %maximum.wave.ptr, align RECIPE_STATE_ALIGN
br label %maximum.wave.skip
maximum.wave.skip:
call void @recipe.local.barrier()
%maximum.group.owner = icmp eq i32 %lid, 0
br i1 %maximum.group.owner, label %maximum.group.loop, label %maximum.group.skip
maximum.group.loop:
%maximum.wave.index = phi i32 [ 0, %maximum.wave.skip ], [ %maximum.wave.index.next, %maximum.group.step ]
%maximum.group.value = phi RECIPE_STATE [ %score.floor, %maximum.wave.skip ], [ %maximum.group.next, %maximum.group.step ]
%maximum.group.more = icmp ult i32 %maximum.wave.index, %waves
br i1 %maximum.group.more, label %maximum.group.step, label %maximum.group.done
maximum.group.step:
%maximum.group.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %tile.RECIPE_STATE, i32 %maximum.wave.index
%maximum.group.wave = load RECIPE_STATE, ptr addrspace(3) %maximum.group.ptr, align RECIPE_STATE_ALIGN
%maximum.group.larger = call i1 @recipe.state.ogt(RECIPE_STATE %maximum.group.wave, RECIPE_STATE %maximum.group.value)
%maximum.group.next = select i1 %maximum.group.larger, RECIPE_STATE %maximum.group.wave, RECIPE_STATE %maximum.group.value
%maximum.wave.index.next = add i32 %maximum.wave.index, 1
br label %maximum.group.loop
maximum.group.done:
store RECIPE_STATE %maximum.group.value, ptr addrspace(3) %maximum.global.ptr, align RECIPE_STATE_ALIGN
br label %maximum.group.skip
maximum.group.skip:
call void @recipe.local.barrier()
%maximum.global = load RECIPE_STATE, ptr addrspace(3) %maximum.global.ptr, align RECIPE_STATE_ALIGN
br label %denominator.loop
denominator.loop:
%denominator.key = phi i32 [ %lid, %maximum.group.skip ], [ %denominator.key.next, %denominator.step ]
%denominator.value = phi RECIPE_STATE [ 0x0000000000000000, %maximum.group.skip ], [ %denominator.next, %denominator.step ]
%denominator.more = icmp ult i32 %denominator.key, %active.limit
br i1 %denominator.more, label %denominator.step, label %denominator.wave.loop
denominator.step:
%denominator.row = mul i32 %head, %length
%denominator.slot = add i32 %denominator.row, %denominator.key
%denominator.slot.wide = zext i32 %denominator.slot to i64
%denominator.score = call RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %denominator.slot.wide)
%denominator.centered = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %denominator.score, RECIPE_STATE %maximum.global)
%denominator.exp = call RECIPE_STATE @recipe.state.exp(RECIPE_STATE %denominator.centered)
%denominator.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %denominator.value, RECIPE_STATE %denominator.exp)
%denominator.key.next = add i32 %denominator.key, %block
br label %denominator.loop
denominator.wave.loop:
%denominator.offset = phi i32 [ %wave.half.start, %denominator.loop ], [ %denominator.offset.next, %denominator.wave.step ]
%denominator.wave.value = phi RECIPE_STATE [ %denominator.value, %denominator.loop ], [ %denominator.wave.value.next, %denominator.wave.step ]
%denominator.wave.more = icmp ne i32 %denominator.offset, 0
br i1 %denominator.wave.more, label %denominator.wave.step, label %denominator.wave.done
denominator.wave.step:
%denominator.partner.lane = xor i32 %lane, %denominator.offset
%denominator.partner.index = mul i32 %denominator.partner.lane, 4
%denominator.partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %denominator.wave.value, i32 %denominator.partner.index)
%denominator.wave.value.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %denominator.wave.value, RECIPE_STATE %denominator.partner)
%denominator.offset.next = lshr i32 %denominator.offset, 1
br label %denominator.wave.loop
denominator.wave.done:
%denominator.owner = icmp eq i32 %lane, 0
br i1 %denominator.owner, label %denominator.wave.store, label %denominator.wave.skip
denominator.wave.store:
%denominator.wave.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %tile.RECIPE_STATE, i32 %wave
store RECIPE_STATE %denominator.wave.value, ptr addrspace(3) %denominator.wave.ptr, align RECIPE_STATE_ALIGN
br label %denominator.wave.skip
denominator.wave.skip:
call void @recipe.local.barrier()
%denominator.group.owner = icmp eq i32 %lid, 0
br i1 %denominator.group.owner, label %denominator.group.loop, label %denominator.group.skip
denominator.group.loop:
%denominator.wave.index = phi i32 [ 0, %denominator.wave.skip ], [ %denominator.wave.index.next, %denominator.group.step ]
%denominator.group.value = phi RECIPE_STATE [ 0x0000000000000000, %denominator.wave.skip ], [ %denominator.group.next, %denominator.group.step ]
%denominator.group.more = icmp ult i32 %denominator.wave.index, %waves
br i1 %denominator.group.more, label %denominator.group.step, label %denominator.group.done
denominator.group.step:
%denominator.group.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) %tile.RECIPE_STATE, i32 %denominator.wave.index
%denominator.group.wave = load RECIPE_STATE, ptr addrspace(3) %denominator.group.ptr, align RECIPE_STATE_ALIGN
%denominator.group.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %denominator.group.value, RECIPE_STATE %denominator.group.wave)
%denominator.wave.index.next = add i32 %denominator.wave.index, 1
br label %denominator.group.loop
denominator.group.done:
store RECIPE_STATE %denominator.group.value, ptr addrspace(3) %denominator.global.ptr, align RECIPE_STATE_ALIGN
br label %denominator.group.skip
denominator.group.skip:
call void @recipe.local.barrier()
%denominator.global = load RECIPE_STATE, ptr addrspace(3) %denominator.global.ptr, align RECIPE_STATE_ALIGN
br label %probability.loop
probability.loop:
%probability.key = phi i32 [ %lid, %denominator.group.skip ], [ %probability.key.next, %probability.step ]
%probability.more = icmp ult i32 %probability.key, %active.limit
br i1 %probability.more, label %probability.step, label %probability.done
probability.step:
%probability.row = mul i32 %head, %length
%probability.slot = add i32 %probability.row, %probability.key
%probability.slot.wide = zext i32 %probability.slot to i64
%probability.score = call RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %probability.slot.wide)
%probability.centered = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %probability.score, RECIPE_STATE %maximum.global)
%probability.exp = call RECIPE_STATE @recipe.state.exp(RECIPE_STATE %probability.centered)
%probability.value = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %probability.exp, RECIPE_STATE %denominator.global)
call void @attention_step_score_store(ptr addrspace(1) %context, i64 %probability.slot.wide, RECIPE_STATE %probability.value)
%probability.key.next = add i32 %probability.key, %block
br label %probability.loop
probability.done:
call void @grid_barrier(i32 %threads)
%round.next = add i32 %round, 1
br label %head.loop
head.done:
br label %output.channel.loop
output.channel.loop:
%output.channel = phi i32 [ %global.wave, %head.done ], [ %output.channel.next, %output.channel.done ]
%output.channel.more = icmp ult i32 %output.channel, %channels
br i1 %output.channel.more, label %output.key.loop, label %output.stats.owner
output.key.loop:
%output.key = phi i32 [ %lane, %output.channel.loop ], [ %output.key.next, %output.key.step ]
%output.sum.0 = phi RECIPE_STATE [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.0.next, %output.key.step ]
%output.sum.1 = phi RECIPE_STATE [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.1.next, %output.key.step ]
%output.sum.2 = phi RECIPE_STATE [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.2.next, %output.key.step ]
%output.sum.3 = phi RECIPE_STATE [ 0x0000000000000000, %output.channel.loop ], [ %output.sum.3.next, %output.key.step ]
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
%output.probability.0.raw = call RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.0.wide)
%output.probability.1.raw = call RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.1.wide)
%output.probability.2.raw = call RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.2.wide)
%output.probability.3.raw = call RECIPE_STATE @attention_step_score_load(ptr addrspace(1) %context, i64 %output.slot.3.wide)
%output.probability.0 = select i1 %output.active.0, RECIPE_STATE %output.probability.0.raw, RECIPE_STATE 0x0000000000000000
%output.probability.1 = select i1 %output.active.1, RECIPE_STATE %output.probability.1.raw, RECIPE_STATE 0x0000000000000000
%output.probability.2 = select i1 %output.active.2, RECIPE_STATE %output.probability.2.raw, RECIPE_STATE 0x0000000000000000
%output.probability.3 = select i1 %output.active.3, RECIPE_STATE %output.probability.3.raw, RECIPE_STATE 0x0000000000000000
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
%output.ptr.0 = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %output.index.0.wide
%output.ptr.1 = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %output.index.1.wide
%output.ptr.2 = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %output.index.2.wide
%output.ptr.3 = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %output.index.3.wide
%output.value.0.raw = load RECIPE_KV, ptr addrspace(1) %output.ptr.0, align RECIPE_KV_ALIGN
%output.value.1.raw = load RECIPE_KV, ptr addrspace(1) %output.ptr.1, align RECIPE_KV_ALIGN
%output.value.2.raw = load RECIPE_KV, ptr addrspace(1) %output.ptr.2, align RECIPE_KV_ALIGN
%output.value.3.raw = load RECIPE_KV, ptr addrspace(1) %output.ptr.3, align RECIPE_KV_ALIGN
%output.value.0.RECIPE_STATE = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %output.value.0.raw)
%output.value.1.RECIPE_STATE = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %output.value.1.raw)
%output.value.2.RECIPE_STATE = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %output.value.2.raw)
%output.value.3.RECIPE_STATE = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %output.value.3.raw)
%output.value.0 = select i1 %output.active.0, RECIPE_STATE %output.value.0.RECIPE_STATE, RECIPE_STATE 0x0000000000000000
%output.value.1 = select i1 %output.active.1, RECIPE_STATE %output.value.1.RECIPE_STATE, RECIPE_STATE 0x0000000000000000
%output.value.2 = select i1 %output.active.2, RECIPE_STATE %output.value.2.RECIPE_STATE, RECIPE_STATE 0x0000000000000000
%output.value.3 = select i1 %output.active.3, RECIPE_STATE %output.value.3.RECIPE_STATE, RECIPE_STATE 0x0000000000000000
%output.sum.0.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %output.sum.0, RECIPE_STATE %output.probability.0, RECIPE_STATE %output.value.0)
%output.sum.1.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %output.sum.1, RECIPE_STATE %output.probability.1, RECIPE_STATE %output.value.1)
%output.sum.2.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %output.sum.2, RECIPE_STATE %output.probability.2, RECIPE_STATE %output.value.2)
%output.sum.3.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %output.sum.3, RECIPE_STATE %output.probability.3, RECIPE_STATE %output.value.3)
%output.key.stride = mul i32 %wave.width, 4
%output.key.next = add i32 %output.key, %output.key.stride
br label %output.key.loop
output.wave.prepare:
%output.sum.01 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %output.sum.0, RECIPE_STATE %output.sum.1)
%output.sum.23 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %output.sum.2, RECIPE_STATE %output.sum.3)
%output.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %output.sum.01, RECIPE_STATE %output.sum.23)
br label %output.wave.loop
output.wave.loop:
%output.offset = phi i32 [ %wave.half.start, %output.wave.prepare ], [ %output.offset.next, %output.wave.step ]
%output.wave.sum = phi RECIPE_STATE [ %output.sum, %output.wave.prepare ], [ %output.wave.sum.next, %output.wave.step ]
%output.wave.more = icmp ne i32 %output.offset, 0
br i1 %output.wave.more, label %output.wave.step, label %output.wave.done
output.wave.step:
%output.partner.lane = xor i32 %lane, %output.offset
%output.partner.index = mul i32 %output.partner.lane, 4
%output.partner = call RECIPE_STATE @recipe.wave.partner(RECIPE_STATE %output.wave.sum, i32 %output.partner.index)
%output.wave.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %output.wave.sum, RECIPE_STATE %output.partner)
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
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.position.phys
%output.half.value = call double @recipe.model.from.state(RECIPE_STATE %output.wave.sum)
store double %output.half.value, ptr addrspace(1) %output.ptr, align 8
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
%flag.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %flag.index
%flag = load double, ptr addrspace(1) %flag.ptr, align 8
%result = call i1 @recipe.ogt(double %flag, double 0.5)
ret i1 %result
}
; The indexer planes live in their own arena, the side projection's output: one
; row holds %index.heads query heads of %index.width channels, then one raw key
; plane of %index.width channels, each channel %length positions long. The
; reference caches the raw key plane, pools a block by mean, then applies the
; key norm and rotary at the block position.
define internal double @attention_index_mean(ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.width, i32 %length, i32 %d) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%model.zero = call double @recipe.encode(RECIPE_STATE %state.zero)
%block.index.wide = zext i32 %block.index to i64 %select.block.wide = zext i32 %select.block to i64 %index.width.wide = zext i32 %index.width to i64 %length.wide = zext i32 %length to i64 %d.wide = zext i32 %d to i64
%start = mul i64 %block.index.wide, %select.block.wide
%query.block = udiv i32 %query, %select.block
%own = icmp eq i32 %block.index, %query.block
%query.next = add i32 %query, 1
%query.next.wide = zext i32 %query.next to i64
%remaining = sub i64 %length.wide, %start
%full = icmp ult i64 %remaining, %select.block.wide
%remaining.i32 = trunc i64 %remaining to i32 %cached.count = select i1 %full, i32 %remaining.i32, i32 %select.block
%count = select i1 %own, i32 %query.next, i32 %cached.count
%start.i32 = trunc i64 %start to i32 %count.less = select i1 %own, i32 %start.i32, i32 0
%filled = sub i32 %count, %count.less
%d.offset = mul i64 %d.wide, %length.wide
br i1 %own, label %prefix.loop, label %cached
cached:
%cached.block = mul i64 %block.index.wide, %index.width.wide
%cached.row = add i64 %representative.start, %cached.block
%cached.index = add i64 %cached.row, %d.wide
%cached.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %cached.index
%cached.value = load double, ptr addrspace(1) %cached.ptr, align 8
br label %done
prefix.loop:
%key = phi i64 [ %start, %entry ], [ %key.next, %prefix.step ]
%sum = phi double [ %model.zero, %entry ], [ %sum.next, %prefix.step ]
%more = icmp ult i64 %key, %query.next.wide
br i1 %more, label %prefix.step, label %done
prefix.step:
%key.position = add i64 %key.origin, %key
%key.index = add i64 %key.position, %d.offset
%key.ptr = getelementptr inbounds double, ptr addrspace(1) %indexer, i64 %key.index
%key.value = load double, ptr addrspace(1) %key.ptr, align 8
%sum.next = call double @recipe.add(double %sum, double %key.value)
%key.next = add i64 %key, 1
br label %prefix.loop
done:
%total = phi double [ %cached.value, %cached ], [ %sum, %prefix.loop ]
%filled.value = call double @recipe.from.u32(i32 %filled)
%mean = call double @recipe.div(double %total, double %filled.value)
ret double %mean
}
; Normalize one indexed key dimension. A trained score uses the deferred key
; scale from the normalization node (the query scales occupy the first
; index.heads*width entries), and receives a raw pooled mean. The unscored path
; returns its cached mean because the key plane was normalized before pooling.
define internal double @attention_index_normalized(ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) nocapture readonly %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.heads, i32 %index.width, i32 %length, i32 %mode, double %epsilon, i1 %pooled, i32 %d) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
br i1 %pooled, label %sum.loop, label %plain
plain:
%plain.mean = call double @attention_index_mean(ptr addrspace(1) %indexer, ptr addrspace(1) %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.width, i32 %length, i32 %d)
ret double %plain.mean
sum.loop:
%sum.d = phi i32 [ 0, %entry ], [ %sum.d.next, %sum.step ]
%sum = phi RECIPE_STATE [ %state.zero, %entry ], [ %sum.next, %sum.step ]
%sum.more = icmp ult i32 %sum.d, %index.width
br i1 %sum.more, label %sum.step, label %sum.done
sum.step:
%sum.mean = call double @attention_index_mean(ptr addrspace(1) %indexer, ptr addrspace(1) %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.width, i32 %length, i32 %sum.d)
%sum.wide = call RECIPE_STATE @recipe.decode(double %sum.mean)
%sum.square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %sum.wide, RECIPE_STATE %sum.wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %sum.square)
%sum.d.next = add i32 %sum.d, 1
br label %sum.loop
sum.done:
%mode.rms = icmp eq i32 %mode, 2
br i1 %mode.rms, label %rms.deviation, label %l2.deviation
rms.deviation:
%width.value = call RECIPE_STATE @recipe.state.from.u32(i32 %index.width)
%variance = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %sum, RECIPE_STATE %width.value)
%epsilon.wide = call RECIPE_STATE @recipe.decode(double %epsilon)
%adjusted = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %variance, RECIPE_STATE %epsilon.wide)
%deviation = call RECIPE_STATE @recipe.state.sqrt(RECIPE_STATE %adjusted)
br label %deviation.merge
l2.deviation:
%norm = call RECIPE_STATE @recipe.state.sqrt(RECIPE_STATE %sum)
%epsilon.l2 = call RECIPE_STATE @recipe.decode(double %epsilon)
%norm.model = call double @recipe.encode(RECIPE_STATE %norm)
%epsilon.l2.model = call double @recipe.encode(RECIPE_STATE %epsilon.l2)
%above = call i1 @recipe.ogt(double %norm.model, double %epsilon.l2.model)
%deviation.l2 = select i1 %above, RECIPE_STATE %norm, RECIPE_STATE %epsilon.l2
br label %deviation.merge
deviation.merge:
%deviation.final = phi RECIPE_STATE [ %deviation, %rms.deviation ], [ %deviation.l2, %l2.deviation ]
%mean.final = call double @attention_index_mean(ptr addrspace(1) %indexer, ptr addrspace(1) %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.width, i32 %length, i32 %d)
%mean.wide = call RECIPE_STATE @recipe.decode(double %mean.final)
%unit = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %mean.wide, RECIPE_STATE %deviation.final)
%mode.scale = icmp eq i32 %mode, 2
br i1 %mode.scale, label %scale.key, label %unit.plain
scale.key:
%index.heads.wide = zext i32 %index.heads to i64 %index.width.wide = zext i32 %index.width to i64 %d.wide = zext i32 %d to i64 %query.scale.count = mul i64 %index.heads.wide, %index.width.wide
%key.scale.index = add i64 %query.scale.count, %d.wide
%key.scale.ptr = getelementptr inbounds double, ptr addrspace(1) %key.weights, i64 %key.scale.index
%key.scale.value = load double, ptr addrspace(1) %key.scale.ptr, align 8
%key.scale.wide = call RECIPE_STATE @recipe.decode(double %key.scale.value)
%scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %unit, RECIPE_STATE %key.scale.wide)
%scaled.model = call double @recipe.encode(RECIPE_STATE %scaled)
br label %unit.merge
unit.plain:
%unit.model = call double @recipe.encode(RECIPE_STATE %unit)
br label %unit.merge
unit.merge:
%result = phi double [ %scaled.model, %scale.key ], [ %unit.model, %unit.plain ]
ret double %result
}
; Rotate one normalized pooled key dimension at its block position. This is the
; same half-width pairing as rope_body, with the block start as position.
define internal double @attention_index_rotated(ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) nocapture readonly %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.heads, i32 %index.width, i32 %length, i32 %mode, i32 %dims, RECIPE_STATE %base, double %epsilon, i1 %pooled, i32 %d) #1 { entry:
%value = call double @attention_index_normalized(ptr addrspace(1) %indexer, ptr addrspace(1) %key.weights, ptr addrspace(1) %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.heads, i32 %index.width, i32 %length, i32 %mode, double %epsilon, i1 %pooled, i32 %d)
%half = udiv i32 %dims, 2
%inside = icmp ult i32 %d, %dims
br i1 %inside, label %rotate, label %finish
rotate:
%upper = icmp uge i32 %d, %half
br i1 %upper, label %rotate.upper, label %rotate.lower
rotate.lower:
%lower.partner = add i32 %d, %half
br label %partner.load
rotate.upper:
%upper.partner = sub i32 %d, %half
br label %partner.load
partner.load:
%partner.d = phi i32 [ %lower.partner, %rotate.lower ], [ %upper.partner, %rotate.upper ]
%half.index = phi i32 [ %d, %rotate.lower ], [ %upper.partner, %rotate.upper ]
%partner.value = call double @attention_index_normalized(ptr addrspace(1) %indexer, ptr addrspace(1) %key.weights, ptr addrspace(1) %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.heads, i32 %index.width, i32 %length, i32 %mode, double %epsilon, i1 %pooled, i32 %partner.d)
%two.index = mul i32 %half.index, 2
%two.value = call RECIPE_STATE @recipe.state.from.u32(i32 %two.index)
%dims.value = call RECIPE_STATE @recipe.state.from.u32(i32 %dims)
%ratio = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %two.value, RECIPE_STATE %dims.value)
%log.base = call RECIPE_STATE @recipe.state.log(RECIPE_STATE %base)
%exponent.positive = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %ratio, RECIPE_STATE %log.base)
%exponent = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %exponent.positive)
%frequency = call RECIPE_STATE @recipe.state.exp(RECIPE_STATE %exponent)
%position = mul i32 %block.index, %select.block
%position.value = call RECIPE_STATE @recipe.state.from.u32(i32 %position)
%angle = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %position.value, RECIPE_STATE %frequency)
%cos = call RECIPE_STATE @recipe.state.cos(RECIPE_STATE %angle)
%sin = call RECIPE_STATE @recipe.state.sin(RECIPE_STATE %angle)
%value.wide = call RECIPE_STATE @recipe.decode(double %value)
%partner.wide = call RECIPE_STATE @recipe.decode(double %partner.value)
%cos.part = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %value.wide, RECIPE_STATE %cos)
%sin.magnitude = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %partner.wide, RECIPE_STATE %sin)
br i1 %upper, label %rotate.upper.result, label %rotate.lower.result
rotate.lower.result:
%lower.result = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %cos.part, RECIPE_STATE %sin.magnitude)
br label %rotate.result
rotate.upper.result:
%upper.result = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %cos.part, RECIPE_STATE %sin.magnitude)
br label %rotate.result
rotate.result:
%rotated = phi RECIPE_STATE [ %lower.result, %rotate.lower.result ], [ %upper.result, %rotate.upper.result ]
%rotated.model = call double @recipe.encode(RECIPE_STATE %rotated)
br label %finish
finish:
%result = phi double [ %value, %entry ], [ %rotated.model, %rotate.result ]
ret double %result
}
; Compatibility wrapper retained for callers that need one transformed
; representative dimension.
define internal double @attention_index_representative(ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) nocapture readonly %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.heads, i32 %index.width, i32 %length, i32 %mode, i32 %dims, RECIPE_STATE %base, double %epsilon, i1 %pooled, i32 %d) #1 { entry:
%result = call double @attention_index_rotated(ptr addrspace(1) %indexer, ptr addrspace(1) %key.weights, ptr addrspace(1) %context,
i64 %key.origin, i64 %representative.start, i32 %query, i32 %block.index, i32 %select.block, i32 %index.heads, i32 %index.width, i32 %length, i32 %mode, i32 %dims, RECIPE_STATE %base, double %epsilon, i1 %pooled, i32 %d)
ret double %result
}
; The running sum of indexer keys of one key block, extended by the keys of the
; block that lie in the forward window `begin..end`. A block whose first key
; lies in the window starts from zero; one that began earlier keeps the sum an
; earlier window left, so a decode step adds one key to one block.
define internal void @attention_index_body( ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) %context,
i64 %p, i32 %begin, i32 %end, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width,
i32 %select.block, i1 %gate, double %epsilon, i32 %index.mode, i32 %index.dims, i1 %index.pooled, RECIPE_STATE %index.base ) #1 { entry:
%from.wide = zext i32 %from to i64 %channels.wide = zext i32 %channels to i64 %rows.wide = zext i32 %rows to i64 %heads.wide = zext i32 %heads to i64 %index.heads.wide = zext i32 %index.heads to i64 %index.width.wide = zext i32 %index.width to i64 %select.block.wide = zext i32 %select.block to i64 %begin.wide = zext i32 %begin to i64 %end.wide = zext i32 %end to i64
%length = udiv i64 %from.wide, %channels.wide
%index.query.channels = mul i64 %index.heads.wide, %index.width.wide
%index.channels = add i64 %index.query.channels, %index.width.wide
%row.stride = mul i64 %index.channels, %length
%index.key.base = mul i64 %index.query.channels, %length
%blocks.numerator = add i64 %length, %select.block.wide
%blocks.less = sub i64 %blocks.numerator, 1
%blocks = udiv i64 %blocks.less, %select.block.wide
%statistics.rows = mul i64 %rows.wide, %heads.wide
%statistics.plane = mul i64 %statistics.rows, %length
%representative.base = mul i64 %statistics.plane, 2
%representative.stride = mul i64 %blocks, %index.width.wide
%row = udiv i64 %p, %blocks
%block.index = urem i64 %p, %blocks
%row.base = mul i64 %row, %row.stride
%key.origin = add i64 %row.base, %index.key.base
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
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%model.zero = call double @recipe.encode(RECIPE_STATE %state.zero)
br i1 %clear, label %clear.loop, label %key.loop
clear.loop:
%clear.d = phi i32 [ 0, %entry ], [ %clear.next, %clear.step ]
%clear.more = icmp ult i32 %clear.d, %index.width
br i1 %clear.more, label %clear.step, label %key.loop
clear.step:
%clear.d.wide = zext i32 %clear.d to i64
%clear.index = add i64 %representative.start, %clear.d.wide
%clear.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %clear.index
store double %model.zero, ptr addrspace(1) %clear.ptr, align 8
%clear.next = add i32 %clear.d, 1
br label %clear.loop
key.loop:
%key = phi i64 [ %first, %entry ], [ %first, %clear.loop ], [ %key.advance, %key.step ]
%key.more = icmp ult i64 %key, %stop
br i1 %key.more, label %key.prepare, label %exit
key.prepare:
%key.position = add i64 %key.origin, %key
br label %dim.loop
dim.loop:
%dim = phi i32 [ 0, %key.prepare ], [ %dim.advance, %dim.step ]
%dim.more = icmp ult i32 %dim, %index.width
br i1 %dim.more, label %dim.step, label %key.step
dim.step:
%dim.wide = zext i32 %dim to i64 %dim.offset = mul i64 %dim.wide, %length
%dim.index = add i64 %key.position, %dim.offset
%dim.ptr = getelementptr inbounds double, ptr addrspace(1) %indexer, i64 %dim.index
%dim.value = load double, ptr addrspace(1) %dim.ptr, align 8
%dim.target = add i64 %representative.start, %dim.wide
%dim.target.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %dim.target
%dim.prior = load double, ptr addrspace(1) %dim.target.ptr, align 8
%dim.sum = call double @recipe.add(double %dim.prior, double %dim.value)
store double %dim.sum, ptr addrspace(1) %dim.target.ptr, align 8
%dim.advance = add i32 %dim, 1
br label %dim.loop
key.step:
%key.advance = add i64 %key, 1
br label %key.loop
exit:
ret void
}
; Block scores and the selection threshold of one query. Every indexer query
; head scores every causal block representative, the heads' scores add, and
; the threshold is the score of the keep-th best, so a query keeps every block
; whose score reaches it.
; Block %block's score for query %p: the sum over indexer heads of the rectified
; dot of the head's query with the block representative, rounded to the model
; type after each head as the whole-row body stores it.
define internal void @attention_select_score_body( ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) %context,
i64 %p, i32 %keep, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads, i32 %value.heads, i32 %index.heads,
i32 %index.width, i32 %select.block, i1 %gate, double %epsilon, i32 %index.mode, i32 %index.dims, i1 %index.pooled, RECIPE_STATE %index.base, i32 %block ) #1 { entry:
%from.wide = zext i32 %from to i64 %channels.wide = zext i32 %channels to i64 %rows.wide = zext i32 %rows to i64 %heads.wide = zext i32 %heads to i64 %index.heads.wide = zext i32 %index.heads to i64 %index.width.wide = zext i32 %index.width to i64 %select.block.wide = zext i32 %select.block to i64
%length = udiv i64 %from.wide, %channels.wide
%index.query.channels = mul i64 %index.heads.wide, %index.width.wide
%index.channels = add i64 %index.query.channels, %index.width.wide
%row.stride = mul i64 %index.channels, %length
%index.key.base = mul i64 %index.query.channels, %length
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
%row.base = mul i64 %row, %row.stride
%key.origin = add i64 %row.base, %index.key.base
%query.position = add i64 %row.base, %query
%count.less = udiv i64 %query, %select.block.wide
%count.wide = add i64 %count.less, 1
%count = trunc i64 %count.wide to i32
%score.query = mul i64 %p, %score.stride
%score.start = add i64 %score.base, %score.query
%representative.row = mul i64 %row, %representative.stride
%representative.start = add i64 %representative.base, %representative.row
%query.i32 = trunc i64 %query to i32 %length.i32 = trunc i64 %length to i32
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%model.zero = call double @recipe.encode(RECIPE_STATE %state.zero)
%in.range = icmp ult i32 %block, %count
br i1 %in.range, label %head.loop, label %exit
head.loop:
%head = phi i32 [ 0, %entry ], [ %head.advance, %head.done ]
%total = phi double [ %model.zero, %entry ], [ %total.next, %head.done ]
%head.more = icmp ult i32 %head, %index.heads
br i1 %head.more, label %head.prepare, label %store
head.prepare:
%head.wide = zext i32 %head to i64 %head.offset = mul i64 %head.wide, %index.width.wide
%head.plane = mul i64 %head.offset, %length
%head.base = add i64 %query.position, %head.plane
br label %dim.loop
dim.loop:
%d = phi i32 [ 0, %head.prepare ], [ %d.advance, %dim.step ]
%sum = phi RECIPE_STATE [ %state.zero, %head.prepare ], [ %sum.next, %dim.step ]
%dim.more = icmp ult i32 %d, %index.width
br i1 %dim.more, label %dim.step, label %head.done
dim.step:
%d.wide = zext i32 %d to i64 %dim.offset = mul i64 %d.wide, %length
%query.index = add i64 %head.base, %dim.offset
%query.ptr = getelementptr inbounds double, ptr addrspace(1) %indexer, i64 %query.index
%query.value = load double, ptr addrspace(1) %query.ptr, align 8
%representative.value = call double @attention_index_representative(ptr addrspace(1) %indexer, ptr addrspace(1) %key.weights, ptr addrspace(1) %context,
i64 %key.origin, i64 %representative.start, i32 %query.i32, i32 %block, i32 %select.block, i32 %index.heads, i32 %index.width, i32 %length.i32, i32 %index.mode, i32 %index.dims, RECIPE_STATE %index.base, double %epsilon, i1 %index.pooled, i32 %d)
%query.wide = call RECIPE_STATE @recipe.decode(double %query.value)
%representative.wide = call RECIPE_STATE @recipe.decode(double %representative.value)
%term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %query.wide, RECIPE_STATE %representative.wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %term)
%d.advance = add i32 %d, 1
br label %dim.loop
head.done:
%sum.model = call double @recipe.encode(RECIPE_STATE %sum)
%positive = call i1 @recipe.ogt(double %sum.model, double 0.0)
%head.model = select i1 %positive, double %sum.model, double %model.zero
%head.state = call RECIPE_STATE @recipe.decode(double %head.model)
%total.state = call RECIPE_STATE @recipe.decode(double %total)
%total.added = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %total.state, RECIPE_STATE %head.state)
%total.next = call double @recipe.encode(RECIPE_STATE %total.added)
%head.advance = add i32 %head, 1
br label %head.loop
store:
%block.wide = zext i32 %block to i64 %score.index = add i64 %score.start, %block.wide
%score.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %score.index
store double %total, ptr addrspace(1) %score.ptr, align 8
br label %exit
exit:
ret void
}
; Block %block is kept for query %p while fewer than %keep blocks score above it,
; an equal score ranking the earlier block first.
define internal void @attention_select_rank_body( ptr addrspace(1) nocapture readonly %indexer, ptr addrspace(1) nocapture readonly %key.weights, ptr addrspace(1) %context,
i64 %p, i32 %keep, i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %kv.heads, i32 %value.heads, i32 %index.heads,
i32 %index.width, i32 %select.block, i1 %gate, double %epsilon, i32 %index.mode, i32 %index.dims, i1 %index.pooled, RECIPE_STATE %index.base, i32 %block ) #1 { entry:
%from.wide = zext i32 %from to i64 %channels.wide = zext i32 %channels to i64 %rows.wide = zext i32 %rows to i64 %heads.wide = zext i32 %heads to i64 %index.heads.wide = zext i32 %index.heads to i64 %index.width.wide = zext i32 %index.width to i64 %select.block.wide = zext i32 %select.block to i64
%length = udiv i64 %from.wide, %channels.wide
%index.query.channels = mul i64 %index.heads.wide, %index.width.wide
%index.channels = add i64 %index.query.channels, %index.width.wide
%row.stride = mul i64 %index.channels, %length
%index.key.base = mul i64 %index.query.channels, %length
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
%row.base = mul i64 %row, %row.stride
%key.origin = add i64 %row.base, %index.key.base
%query.position = add i64 %row.base, %query
%count.less = udiv i64 %query, %select.block.wide
%count.wide = add i64 %count.less, 1
%count = trunc i64 %count.wide to i32
%score.query = mul i64 %p, %score.stride
%score.start = add i64 %score.base, %score.query
%representative.row = mul i64 %row, %representative.stride
%representative.start = add i64 %representative.base, %representative.row
%query.i32 = trunc i64 %query to i32 %length.i32 = trunc i64 %length to i32
%in.range = icmp ult i32 %block, %count
br i1 %in.range, label %rank.body, label %exit
rank.body:
%block.wide = zext i32 %block to i64 %own.index = add i64 %score.start, %block.wide
%own.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %own.index
%own = load double, ptr addrspace(1) %own.ptr, align 8
br label %rank.loop
rank.loop:
%c = phi i32 [ 0, %rank.body ], [ %c.next, %rank.step ]
%ahead = phi i32 [ 0, %rank.body ], [ %ahead.next, %rank.step ]
%more = icmp ult i32 %c, %count
br i1 %more, label %rank.step, label %rank.decide
rank.step:
%c.wide = zext i32 %c to i64 %c.index = add i64 %score.start, %c.wide
%c.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %c.index
%c.score = load double, ptr addrspace(1) %c.ptr, align 8
%greater = call i1 @recipe.ogt(double %c.score, double %own)
%same = call i1 @recipe.oeq(double %c.score, double %own)
%earlier = icmp ult i32 %c, %block
%tie = and i1 %same, %earlier
%before = or i1 %greater, %tie
%one = zext i1 %before to i32
%ahead.next = add i32 %ahead, %one
%c.next = add i32 %c, 1
br label %rank.loop
rank.decide:
%admit = icmp ult i32 %ahead, %keep
%flag = call double @recipe.from.u1(i1 %admit)
%flag.base = add i64 %score.start, %blocks
%flag.index = add i64 %flag.base, %block.wide
%flag.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %flag.index
store double %flag, ptr addrspace(1) %flag.ptr, align 8
br label %exit
exit:
ret void
}

define internal void @attention_tile_products(ptr addrspace(1) nocapture readonly %output, i64 %output.row,
i32 %delta.base, i32 %product.base, i32 %query.base, i32 %query.count, i32 %head.start,
i32 %head.width, i32 %length, i32 %lid, i32 %block) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%output.row.wide = add i64 %output.row, 0 %delta.base.wide = zext i32 %delta.base to i64 %product.base.wide = zext i32 %product.base to i64 %query.base.wide = zext i32 %query.base to i64 %head.start.wide = zext i32 %head.start to i64 %head.width.wide = zext i32 %head.width to i64 %length.wide = zext i32 %length to i64
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
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
%sum = phi RECIPE_STATE [ %zero, %prepare ], [ %sum.next, %channel.step ]
%channel.more = icmp ult i32 %channel, %head.width
br i1 %channel.more, label %channel.step, label %store
channel.step:
%channel.wide = zext i32 %channel to i64 %shared.row = mul i64 %query.wide, %head.width.wide
%shared.local = add i64 %shared.row, %channel.wide
%delta.index = add i64 %delta.base.wide, %shared.local
%delta.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i64 %delta.index
%delta = load RECIPE_STATE, ptr addrspace(3) %delta.ptr, align RECIPE_STATE_ALIGN
%output.channel = add i64 %head.start.wide, %channel.wide
%output.channel.base = mul i64 %output.channel, %length.wide
%position = add i64 %query.base.wide, %query.wide
%output.local = add i64 %output.channel.base, %position
%output.index = add i64 %output.row.wide, %output.local
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.index
%output.value.model = load double, ptr addrspace(1) %output.ptr, align 8
%output.value = call RECIPE_STATE @recipe.decode(double %output.value.model)
%delta.wide = select i1 true, RECIPE_STATE %delta, RECIPE_STATE %delta
%output.wide = select i1 true, RECIPE_STATE %output.value, RECIPE_STATE %output.value
%term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %delta.wide, RECIPE_STATE %output.wide)
%sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %sum, RECIPE_STATE %term)
%channel.next = add i32 %channel, 1
br label %channel.loop
store:
%product.index = add i64 %product.base.wide, %query.wide
%product.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i64 %product.index
store RECIPE_STATE %sum, ptr addrspace(3) %product.ptr, align RECIPE_STATE_ALIGN
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
double %scale, i32 %lid, i32 %block, i64 %score.row, i32 %blocks, i32 %select.block, i1 %select) #1 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%query.shared.wide = zext i32 %query.shared to i64 %probability.shared.wide = zext i32 %probability.shared to i64 %derivative.shared.wide = zext i32 %derivative.shared to i64 %product.shared.wide = zext i32 %product.shared to i64 %query.base.wide = zext i32 %query.base to i64 %key.base.wide = zext i32 %key.base to i64 %tile.n.wide = zext i32 %tile.n to i64 %head.job.wide = add i64 %head.job, 0 %length.wide = add i64 %length, 0 %statistics.denominator.base.wide = add i64 %statistics.denominator.base, 0 %head.width.wide = zext i32 %head.width to i64
%pair.count = mul i32 %query.count, %key.count
%zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
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
%score.raw = call RECIPE_STATE @attention_tile_dot_state(i32 %query.local, i32 %key.local, i32 %head.width, i32 %query.shared, i32 %key.shared)
%scale.wide = call RECIPE_STATE @recipe.decode(double %scale)
%score = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %score.raw, RECIPE_STATE %scale.wide)
%dp = call RECIPE_STATE @attention_tile_dot_state(i32 %query.local, i32 %key.local, i32 %head.width, i32 %delta.shared, i32 %value.shared)
%statistics.base = mul i64 %head.job.wide, %length.wide
%statistics.index = add i64 %statistics.base, %query
%maximum.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %statistics.index
%maximum = load double, ptr addrspace(1) %maximum.ptr, align 8
%denominator.index = add i64 %statistics.denominator.base.wide, %statistics.index
%denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %denominator.index
%denominator = load double, ptr addrspace(1) %denominator.ptr, align 8
%maximum.wide = call RECIPE_STATE @recipe.decode(double %maximum)
%denominator.wide = call RECIPE_STATE @recipe.decode(double %denominator)
%centered = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %score, RECIPE_STATE %maximum.wide)
%exponential = call RECIPE_STATE @recipe.state.exp(RECIPE_STATE %centered)
%probability.wide = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %exponential, RECIPE_STATE %denominator.wide)
%product.index = add i64 %product.shared.wide, %query.local.wide
%product.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i64 %product.index
%product = load RECIPE_STATE, ptr addrspace(3) %product.ptr, align RECIPE_STATE_ALIGN
%product.wide = select i1 true, RECIPE_STATE %product, RECIPE_STATE %product
%dp.centered = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %dp, RECIPE_STATE %product.wide)
%derivative.wide = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %probability.wide, RECIPE_STATE %dp.centered)
br label %store
invalid:
br label %store
store:
%probability.value = phi RECIPE_STATE [ %probability.wide, %complete ], [ %zero, %invalid ]
%derivative.value = phi RECIPE_STATE [ %derivative.wide, %complete ], [ %zero, %invalid ]
%pair.row = mul i64 %query.local.wide, %tile.n.wide
%pair.local = add i64 %pair.row, %key.local.wide
%probability.index = add i64 %probability.shared.wide, %pair.local
%probability.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i64 %probability.index
store RECIPE_STATE %probability.value, ptr addrspace(3) %probability.ptr, align RECIPE_STATE_ALIGN
%derivative.index = add i64 %derivative.shared.wide, %pair.local
%derivative.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i64 %derivative.index
store RECIPE_STATE %derivative.value, ptr addrspace(3) %derivative.ptr, align RECIPE_STATE_ALIGN
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
%q16 = alloca [256 x RECIPE_KV], align RECIPE_KV_ALIGN, addrspace(5)
%v16 = alloca [256 x RECIPE_KV], align RECIPE_KV_ALIGN, addrspace(5)
%lanes = alloca [64 x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5)
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%kv.group = udiv i32 %heads, %kv.heads
%value.group = udiv i32 %heads, %value.heads
%kv.head = udiv i32 %head, %kv.group
%value.head = udiv i32 %head, %value.group
%kv.channels = mul i32 %kv.heads, %width
%kv.plane = mul i32 %kv.channels, %length
%kv.plane.wide = zext i32 %kv.plane to i64
%length.wide = zext i32 %length to i64
%position.wide = zext i32 %position to i64
%width.state = call RECIPE_STATE @recipe.state.from.u32(i32 %width)
%root = call RECIPE_STATE @recipe.state.sqrt(RECIPE_STATE %width.state)
%q.scale.default = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %one, RECIPE_STATE %root)
%q.scale = select i1 %unscaled, RECIPE_STATE %one, RECIPE_STATE %q.scale.default
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
%zero.kv = call RECIPE_KV @recipe.kv.from.state(RECIPE_STATE %state.zero)
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
%q.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %q.index.phys
%q.model = load double, ptr addrspace(1) %q.ptr, align 8
%q.value = call RECIPE_STATE @recipe.state.from.model(double %q.model)
%q.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %q.value, RECIPE_STATE %q.scale)
%q.kv = call RECIPE_KV @recipe.kv.from.state(RECIPE_STATE %q.scaled)
%q16.ptr = getelementptr RECIPE_KV, ptr addrspace(5) %q16, i32 %q.d
store RECIPE_KV %q.kv, ptr addrspace(5) %q16.ptr, align RECIPE_KV_ALIGN
%v16.init.ptr = getelementptr RECIPE_KV, ptr addrspace(5) %v16, i32 %q.d
store RECIPE_KV %zero.kv, ptr addrspace(5) %v16.init.ptr, align RECIPE_KV_ALIGN
%q.d.next = add i32 %q.d, 1
br label %q.loop
q.done:
%chunks = lshr i32 %width, 6
%np = shl i32 %chunks, 6
; Minus infinity as the state computes it: -1 / 0, so no literal names a type.
%minus.one = call RECIPE_STATE @recipe.state.neg(RECIPE_STATE %one)
%negative.infinity = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %minus.one, RECIPE_STATE %state.zero)
br label %key.loop
key.loop:
%key = phi i32 [ 0, %q.done ], [ %key.next, %key.finish ]
%m = phi RECIPE_STATE [ %negative.infinity, %q.done ], [ %m.next, %key.finish ]
%s.sum = phi RECIPE_STATE [ %state.zero, %q.done ], [ %s.sum.next, %key.finish ]
%key.more = icmp ule i32 %key, %position
br i1 %key.more, label %lane.zero.loop, label %key.done
lane.zero.loop:
%lz = phi i32 [ 0, %key.loop ], [ %lz.next, %lane.zero.step ]
%lz.more = icmp ult i32 %lz, 64
br i1 %lz.more, label %lane.zero.step, label %chunk.loop
lane.zero.step:
%lz.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %lz
store RECIPE_STATE %state.zero, ptr addrspace(5) %lz.ptr, align RECIPE_STATE_ALIGN
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
%k.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %k.index
%k.kv = load RECIPE_KV, ptr addrspace(1) %k.ptr, align RECIPE_KV_ALIGN
%k.value = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %k.kv)
%q16.lane.ptr = getelementptr RECIPE_KV, ptr addrspace(5) %q16, i32 %e
%q.lane.kv = load RECIPE_KV, ptr addrspace(5) %q16.lane.ptr, align RECIPE_KV_ALIGN
%q.lane.value = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %q.lane.kv)
%lane.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %lane
%lane.value = load RECIPE_STATE, ptr addrspace(5) %lane.ptr, align RECIPE_STATE_ALIGN
%lane.value.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %lane.value, RECIPE_STATE %k.value, RECIPE_STATE %q.lane.value)
store RECIPE_STATE %lane.value.next, ptr addrspace(5) %lane.ptr, align RECIPE_STATE_ALIGN
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
%fold.x0.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %fold
%fold.x1.index = add i32 %fold, 16
%fold.x1.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %fold.x1.index
%fold.x2.index = add i32 %fold, 32
%fold.x2.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %fold.x2.index
%fold.x3.index = add i32 %fold, 48
%fold.x3.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %fold.x3.index
%fold.x0 = load RECIPE_STATE, ptr addrspace(5) %fold.x0.ptr, align RECIPE_STATE_ALIGN
%fold.x1 = load RECIPE_STATE, ptr addrspace(5) %fold.x1.ptr, align RECIPE_STATE_ALIGN
%fold.x2 = load RECIPE_STATE, ptr addrspace(5) %fold.x2.ptr, align RECIPE_STATE_ALIGN
%fold.x3 = load RECIPE_STATE, ptr addrspace(5) %fold.x3.ptr, align RECIPE_STATE_ALIGN
%fold.x02 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %fold.x0, RECIPE_STATE %fold.x2)
%fold.x13 = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %fold.x1, RECIPE_STATE %fold.x3)
%fold.x = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %fold.x02, RECIPE_STATE %fold.x13)
store RECIPE_STATE %fold.x, ptr addrspace(5) %fold.x0.ptr, align RECIPE_STATE_ALIGN
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
%half.low.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %half.lane
%half.high.index = add i32 %half.lane, %half
%half.high.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 %half.high.index
%half.low = load RECIPE_STATE, ptr addrspace(5) %half.low.ptr, align RECIPE_STATE_ALIGN
%half.high = load RECIPE_STATE, ptr addrspace(5) %half.high.ptr, align RECIPE_STATE_ALIGN
%half.sum = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %half.low, RECIPE_STATE %half.high)
store RECIPE_STATE %half.sum, ptr addrspace(5) %half.low.ptr, align RECIPE_STATE_ALIGN
%half.lane.next = add i32 %half.lane, 1
br label %half.lane.loop
half.done:
%half.next = lshr i32 %half, 1
br label %half.loop
score.ready:
; The values past the last chunk of 64 add one product at a time.
%rest.ptr = getelementptr RECIPE_STATE, ptr addrspace(5) %lanes, i32 0
%rest.start = load RECIPE_STATE, ptr addrspace(5) %rest.ptr, align RECIPE_STATE_ALIGN
br label %rest.loop
rest.loop:
%rest.e = phi i32 [ %np, %score.ready ], [ %rest.e.next, %rest.step ]
%rest.sum = phi RECIPE_STATE [ %rest.start, %score.ready ], [ %rest.sum.next, %rest.step ]
%rest.more = icmp ult i32 %rest.e, %width
br i1 %rest.more, label %rest.step, label %softmax
rest.step:
%rest.e.wide = zext i32 %rest.e to i64
%rest.offset = mul i64 %rest.e.wide, %length.wide
%rest.k.base = add i64 %k.row, %rest.offset
%rest.key.wide = zext i32 %key to i64
%rest.k.index = add i64 %rest.k.base, %rest.key.wide
%rest.k.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %rest.k.index
%rest.k.kv = load RECIPE_KV, ptr addrspace(1) %rest.k.ptr, align RECIPE_KV_ALIGN
%rest.k = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %rest.k.kv)
%rest.q.ptr = getelementptr RECIPE_KV, ptr addrspace(5) %q16, i32 %rest.e
%rest.q.kv = load RECIPE_KV, ptr addrspace(5) %rest.q.ptr, align RECIPE_KV_ALIGN
%rest.q = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %rest.q.kv)
%rest.product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %rest.k, RECIPE_STATE %rest.q)
%rest.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %rest.sum, RECIPE_STATE %rest.product)
%rest.e.next = add i32 %rest.e, 1
br label %rest.loop
softmax:
%greater = call i1 @recipe.state.ogt(RECIPE_STATE %rest.sum, RECIPE_STATE %m)
%m.next = select i1 %greater, RECIPE_STATE %rest.sum, RECIPE_STATE %m
%m.drop = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %m, RECIPE_STATE %m.next)
%ms.raw = call RECIPE_STATE @recipe.libm.exp(RECIPE_STATE %m.drop)
%ms = select i1 %greater, RECIPE_STATE %ms.raw, RECIPE_STATE %one
%s.drop = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %rest.sum, RECIPE_STATE %m.next)
%vs.raw = call RECIPE_STATE @recipe.libm.exp(RECIPE_STATE %s.drop)
%vs = select i1 %greater, RECIPE_STATE %one, RECIPE_STATE %vs.raw
br label %value.loop
value.loop:
%vd = phi i32 [ 0, %softmax ], [ %vd.next, %value.step ]
%vd.more = icmp ult i32 %vd, %width
br i1 %vd.more, label %value.step, label %key.finish
value.step:
%v16.ptr = getelementptr RECIPE_KV, ptr addrspace(5) %v16, i32 %vd
%acc.kv = load RECIPE_KV, ptr addrspace(5) %v16.ptr, align RECIPE_KV_ALIGN
%acc = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %acc.kv)
%acc.scaled.raw = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %acc, RECIPE_STATE %ms)
%acc.scaled.kv = call RECIPE_KV @recipe.kv.from.state(RECIPE_STATE %acc.scaled.raw)
%acc.scaled = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %acc.scaled.kv)
%acc.base = select i1 %greater, RECIPE_STATE %acc.scaled, RECIPE_STATE %acc
%vd.wide = zext i32 %vd to i64
%vd.offset = mul i64 %vd.wide, %length.wide
%v.index.base = add i64 %v.row, %vd.offset
%v.key.wide = zext i32 %key to i64
%v.index = add i64 %v.index.base, %v.key.wide
%v.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %v.index
%v.kv = load RECIPE_KV, ptr addrspace(1) %v.ptr, align RECIPE_KV_ALIGN
%v.value = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %v.kv)
%acc.next.raw = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %acc.base, RECIPE_STATE %v.value, RECIPE_STATE %vs)
%acc.next.kv = call RECIPE_KV @recipe.kv.from.state(RECIPE_STATE %acc.next.raw)
store RECIPE_KV %acc.next.kv, ptr addrspace(5) %v16.ptr, align RECIPE_KV_ALIGN
%vd.next = add i32 %vd, 1
br label %value.loop
key.finish:
%s.sum.next = call RECIPE_STATE @recipe.state.madd(RECIPE_STATE %vs, RECIPE_STATE %s.sum, RECIPE_STATE %ms)
%key.next = add i32 %key, 1
br label %key.loop
key.done:
%inverse = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %one, RECIPE_STATE %s.sum)
br label %out.loop
out.loop:
%od = phi i32 [ 0, %key.done ], [ %od.next, %out.store ]
%od.more = icmp ult i32 %od, %width
br i1 %od.more, label %out.step, label %done
out.step:
%out.acc.ptr = getelementptr RECIPE_KV, ptr addrspace(5) %v16, i32 %od
%out.acc.kv = load RECIPE_KV, ptr addrspace(5) %out.acc.ptr, align RECIPE_KV_ALIGN
%out.acc = call RECIPE_STATE @recipe.kv.to.state(RECIPE_KV %out.acc.kv)
%out.value = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %out.acc, RECIPE_STATE %inverse)
%od.wide = zext i32 %od to i64
%out.offset = mul i64 %od.wide, %length.wide
%out.local.base = add i64 %q.base, %out.offset
%out.local = add i64 %out.local.base, %position.wide
%out.index = add i64 %output.row, %out.local
%out.model = call double @recipe.model.from.state(RECIPE_STATE %out.value)
br i1 %gate, label %out.gate, label %out.store
out.gate:
%gate.index = add i64 %gate.row, %out.local
%gate.index.phys = call i64 @recipe.window.index(i64 %gate.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%gate.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %gate.index.phys
%gate.value = load double, ptr addrspace(1) %gate.ptr, align 8
%gate.factor = call double @recipe.sigmoid(double %gate.value)
%out.gated = call double @recipe.mul(double %out.model, double %gate.factor)
br label %out.store
out.store:
%out.result = phi double [ %out.model, %out.step ], [ %out.gated, %out.gate ]
%out.index.phys = call i64 @recipe.window.index(i64 %out.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%out.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %out.index.phys
store double %out.result, ptr addrspace(1) %out.ptr, align 8
%od.next = add i32 %od, 1
br label %out.loop
done:
ret void
}
define internal void @attention_forward_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %weights,
ptr addrspace(1) nocapture writeonly %output, ptr addrspace(1) %context, ptr addrspace(1) %kv.context, i1 %carry,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %query.begin, i32 %query.span, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, RECIPE_STATE %index.base, i1 %online, i32 %buffer.length, i32 %buffer.origin ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%head.width.double = call double @recipe.from.u32(i32 %head.width)
%scale.default = call double @recipe.sqrt(double %head.width.double)
%attention.zero = call double @recipe.from.u32(i32 0)
%attention.one = call double @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(double %attention.zero, double %epsilon)
%scale = select i1 %unscaled, double %attention.one, double %scale.default
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
%statistics.plane = mul i32 %head.jobs, %length
%representative.base = mul i32 %statistics.plane, 2
%representative.stride = mul i32 %blocks, %index.width
%representative.total = mul i32 %representative.stride, %rows
%score.base = add i32 %representative.base, %representative.total
%rows.global = zext i32 %rows to i64 %from.global = zext i32 %from to i64 %channels.global = zext i32 %channels to i64 %heads.global = zext i32 %heads to i64 %length.global = zext i32 %length to i64 %head.width.global = zext i32 %head.width to i64 %blocks.global = zext i32 %blocks to i64 %index.width.global = zext i32 %index.width to i64 %block.divisor.global = zext i32 %block.divisor to i64 %kv.heads.global = zext i32 %kv.heads to i64 %index.heads.global = zext i32 %index.heads to i64
%kv.channels.global = mul i64 %kv.heads.global, %head.width.global %kv.plane.global = mul i64 %kv.channels.global, %length.global %value.heads.global = zext i32 %value.heads to i64 %value.channels.global = mul i64 %value.heads.global, %head.width.global %value.plane.global = mul i64 %value.channels.global, %length.global %kv.planes.global = add i64 %kv.plane.global, %value.plane.global %value.plane.base.global = add i64 %from.global, %kv.plane.global
%index.query.channels.global = mul i64 %index.heads.global, %index.width.global %index.channels.global = add i64 %index.query.channels.global, %index.width.global %index.plane.global = mul i64 %index.channels.global, %length.global
%gate.plane.global = select i1 %gate, i64 %from.global, i64 0 %index.query.base.global = add i64 %from.global, %kv.planes.global %gate.base.global = add i64 %index.query.base.global, 0 %row.stride.global = add i64 %gate.base.global, %gate.plane.global
%blocks.numerator.global = add i64 %length.global, %block.divisor.global %blocks.less.global = sub i64 %blocks.numerator.global, 1 %blocks.full.global = udiv i64 %blocks.less.global, %block.divisor.global %blocks.selected.global = select i1 %select, i64 %blocks.full.global, i64 0
%score.stride.global = mul i64 %blocks.selected.global, 2 %head.jobs.global = mul i64 %rows.global, %heads.global %statistics.plane.global = mul i64 %head.jobs.global, %length.global %representative.base.global = mul i64 %statistics.plane.global, 2 %representative.stride.global = mul i64 %blocks.selected.global, %index.width.global %representative.total.global = mul i64 %representative.stride.global, %rows.global %score.base.global = add i64 %representative.base.global, %representative.total.global
%score.base.wide = add i64 %score.base.global, 0
%length.wide = add i64 %length.global, 0
%score.stride.wide = add i64 %score.stride.global, 0
%score.row.stride = mul i64 %length.wide, %score.stride.wide
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
%oc.source.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %oc.source.index.phys
%oc.value = load double, ptr addrspace(1) %oc.source.ptr, align 8
%oc.cache.plane.wide = zext i32 %oc.cache.plane to i64
%oc.cache.base = add i64 %oc.cache.row, %oc.cache.plane.wide
%oc.cache.index = add i64 %oc.cache.base, %oc.local.index
%oc.cache.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %oc.cache.index
%oc.kv = call RECIPE_KV @recipe.kv.encode(double %oc.value)
store RECIPE_KV %oc.kv, ptr addrspace(1) %oc.cache.ptr, align RECIPE_KV_ALIGN
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
%query.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %query.input.index.phys
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
%key.cache.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.cache.index
%key.cache.kv = load RECIPE_KV, ptr addrspace(1) %key.cache.ptr, align RECIPE_KV_ALIGN
%key.cache.value = call double @recipe.kv.decode(RECIPE_KV %key.cache.kv)
%value.cache.index = add i64 %key.cache.row, %kv.plane.global
%value.cache.index.final = add i64 %value.cache.index, %value.input.local
%value.cache.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %value.cache.index.final
%value.cache.kv = load RECIPE_KV, ptr addrspace(1) %value.cache.ptr, align RECIPE_KV_ALIGN
%value.cache.value = call double @recipe.kv.decode(RECIPE_KV %value.cache.kv)
br label %key.loaded
key.source.load:
%key.input.index.phys = call i64 @recipe.window.index(i64 %key.input.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%key.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %key.input.index.phys
%key.source.value = load double, ptr addrspace(1) %key.input.ptr, align 8
%value.row = add i64 %row.base.wide, %value.plane.base.global
%value.input.index = add i64 %value.row, %value.input.local
%value.input.index.phys = call i64 @recipe.window.index(i64 %value.input.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%value.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %value.input.index.phys
%value.source.value = load double, ptr addrspace(1) %value.input.ptr, align 8
br i1 %carry, label %key.cache.store, label %key.loaded.source
key.cache.store:
%key.store.row = mul i64 %row.wide, %kv.planes.global
%key.store.index = add i64 %key.store.row, %key.input.local
%key.store.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %key.store.index
%key.store.kv = call RECIPE_KV @recipe.kv.encode(double %key.source.value)
store RECIPE_KV %key.store.kv, ptr addrspace(1) %key.store.ptr, align RECIPE_KV_ALIGN
%value.store.index = add i64 %key.store.row, %kv.plane.global
%value.store.index.final = add i64 %value.store.index, %value.input.local
%value.store.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i64 %value.store.index.final
%value.store.kv = call RECIPE_KV @recipe.kv.encode(double %value.source.value)
store RECIPE_KV %value.store.kv, ptr addrspace(1) %value.store.ptr, align RECIPE_KV_ALIGN
; A carried key is read back the way the cache holds it, so the window's
; own keys meet the same rounding as the past ones and a prefill gives
; the same bits however it is windowed.
%key.store.value = call double @recipe.kv.decode(RECIPE_KV %key.store.kv)
%value.store.value = call double @recipe.kv.decode(RECIPE_KV %value.store.kv)
br label %key.loaded
key.loaded.source:
br label %key.loaded
key.loaded:
%key.value = phi double [ %key.cache.value, %key.cache.load ], [ %key.source.value, %key.loaded.source ], [ %key.store.value, %key.cache.store ]
%value.value = phi double [ %value.cache.value, %key.cache.load ], [ %value.source.value, %key.loaded.source ], [ %value.store.value, %key.cache.store ]
%key.shared.index = add i32 %key.base.shared, %key.p
%key.shared.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %key.shared.index
store double %key.value, ptr addrspace(3) %key.shared.ptr, align 8
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
br i1 %select, label %score.selection.chosen, label %score.complete
score.selection.chosen:
%score.kept = call i1 @attention_selected(ptr addrspace(1) %context, i64 %score.row.base, i32 %blocks, i32 %select.block, i32 %score.query, i32 %score.key)
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
; A query whose admitted keys all lie in later tiles has no score yet, so its
; maximum is still the initial negative infinity. Centering on zero instead
; keeps its rescale and probabilities at zero rather than exp(-inf - -inf).
%maximum.scored = call i1 @recipe.ogt(double %maximum.value, double 0xFFF0000000000000)
%maximum.safe = select i1 %maximum.scored, double %maximum.value, double 0.0
%maximum.old.centered = call double @recipe.sub(double %maximum.old, double %maximum.safe)
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
%probability.centered = call double @recipe.sub(double %probability.score, double %maximum.safe)
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
%output.statistics.head.job.wide = zext i32 %head.job to i64 %output.statistics.query.wide = zext i32 %output.query to i64 %output.statistics.head.base = mul i64 %output.statistics.head.job.wide, %length.global
%output.statistics.index = add i64 %output.statistics.head.base, %output.statistics.query.wide
%output.statistics.maximum.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %output.statistics.index
store double %output.maximum, ptr addrspace(1) %output.statistics.maximum.ptr, align 8
%output.statistics.denominator.index = add i64 %statistics.plane.global, %output.statistics.index
%output.statistics.denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %output.statistics.denominator.index
store double %output.denominator, ptr addrspace(1) %output.statistics.denominator.ptr, align 8
br label %output.value.store
output.value.store:
%output.channel = add i32 %head.start, %output.channel.local
%output.channel.wide = zext i32 %output.channel to i64 %output.channel.base = mul i64 %output.channel.wide, %length.global
%output.query.wide = zext i32 %output.query to i64 %output.local = add i64 %output.channel.base, %output.query.wide
%output.row.base = mul i64 %row.wide, %from.global
%output.index = add i64 %output.row.base, %output.local
%output.index.phys = call i64 @recipe.window.index(i64 %output.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.index.phys
br i1 %gate, label %output.gate, label %output.plain
output.gate:
%output.gate.row = add i64 %row.base.wide, %gate.base.global
%output.gate.index = add i64 %output.gate.row, %output.local
%output.gate.index.phys = call i64 @recipe.window.index(i64 %output.gate.index, i32 %length, i32 %buffer.length, i32 %buffer.origin)
%output.gate.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %output.gate.index.phys
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
define internal void @attention_cache_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) %kv.context,
i32 %rows, i32 %from, i32 %heads, i32 %kv.heads, i32 %value.heads, i32 %length, i32 %threads ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
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
%source.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %source.index.final
%value = load double, ptr addrspace(1) %source.ptr, align 8
%kv.row = mul i32 %row, %kv.planes
%kv.index.final = add i32 %kv.row, %within
%kv.ptr = getelementptr inbounds RECIPE_KV, ptr addrspace(1) %kv.context, i32 %kv.index.final
%kv.encoded = call RECIPE_KV @recipe.kv.encode(double %value)
store RECIPE_KV %kv.encoded, ptr addrspace(1) %kv.ptr, align RECIPE_KV_ALIGN
%next = add i32 %p, %threads
br label %cache.loop
cache.done:
ret void
}
define internal void @attention_forward_matrix_body(
ptr addrspace(1) nocapture readonly %input, ptr addrspace(1) nocapture readonly %weights,
ptr addrspace(1) nocapture writeonly %output, ptr addrspace(1) %context, ptr addrspace(1) %kv.context, i1 %carry,
i32 %rows, i32 %from, i32 %heads, i32 %channels, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads,
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, RECIPE_STATE %index.base ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%width.double = call double @recipe.from.u32(i32 %head.width)
%scale.default = call double @recipe.sqrt(double %width.double)
%attention.zero = call double @recipe.from.u32(i32 0)
%attention.one = call double @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(double %attention.zero, double %epsilon)
%scale = select i1 %unscaled, double %attention.one, double %scale.default
br i1 %carry, label %attention.cache.entry, label %attention.cache.done
attention.cache.entry:
call void @attention_cache_body( ptr addrspace(1) %input, ptr addrspace(1) %kv.context, i32 %rows, i32 %from, i32 %heads, i32 %kv.heads, i32 %value.heads, i32 %length, i32 %threads )
call void @recipe.local.barrier()
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
%stage.vector.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.vector.index
%stage.vector.value = load <16 x double>, ptr addrspace(1) %stage.vector.ptr, align 8
call void @contraction_stage_column16(<16 x double> %stage.vector.value, i32 %stage.shared.base, i32 %stage.position, i32 %stage.channel.local, i32 %head.width)
%stage.vector.position.next = add i32 %stage.position, 16
br label %attention.forward.matrix.stage.position.advance
attention.forward.matrix.stage.scalar:
%stage.scalar.index = add i64 %stage.input.base, %stage.position.wide
%stage.scalar.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.scalar.index
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
br i1 %matrix.output.more, label %attention.forward.matrix.score.store.check, label %attention.forward.matrix.score.store.done
attention.forward.matrix.score.store.check:
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
%head.job.wide = zext i32 %head.job to i64 %softmax.query.wide = zext i32 %softmax.query to i64 %statistics.base = mul i64 %head.job.wide, %length.global
%statistics.index = add i64 %statistics.base, %softmax.query.wide
%maximum.context.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %statistics.index
store double %maximum, ptr addrspace(1) %maximum.context.ptr, align 8
%denominator.index = add i64 %statistics.rows.global, %statistics.index
%denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %denominator.index
store double %denominator, ptr addrspace(1) %denominator.ptr, align 8
%softmax.query.next = add i32 %softmax.query, %block
br label %attention.forward.matrix.softmax.loop
attention.forward.matrix.softmax.done:
call void @recipe.local.barrier()
call void @attention_matrix_product(ptr addrspace(1) %output, i32 3, i64 %p.base.wide, i64 %v.base.wide, i64 %output.row.global, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
call void @recipe.local.barrier()
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
%m.safe.wide = zext i32 %m.safe to i64 %term.safe.wide = zext i32 %term.safe to i64
%left.direct.row = mul i64 %m.safe.wide, %length.wide
%left.direct.local = add i64 %left.direct.row, %term.safe.wide
%left.transpose.row = mul i64 %term.safe.wide, %length.wide
%left.transpose.local = add i64 %left.transpose.row, %m.safe.wide
%left.local = select i1 %direct, i64 %left.direct.local, i64 %left.transpose.local
%left.index = add i64 %left.base, %left.local
%left.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i64 0, i64 %left.index
%left.loaded = load double, ptr addrspace(3) %left.ptr, align 8
%head.width.wide = zext i32 %head.width to i64 %n.safe.wide = zext i32 %n.safe to i64 %right.row = mul i64 %term.safe.wide, %head.width.wide
%right.local = add i64 %right.row, %n.safe.wide
%right.index = add i64 %right.base, %right.local
%right.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i64 0, i64 %right.index
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
br i1 %output.more, label %attention.matrix.gradient.store.check, label %attention.matrix.gradient.store.done
attention.matrix.gradient.store.check:
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
%mode.wide = zext i32 %mode to i64 %from.wide = zext i32 %from to i64 %output.plane.raw = mul i64 %mode.wide, %from.wide
%output.plane = select i1 %forward, i64 0, i64 %output.plane.raw
%output.row = add i64 %row.base, %output.plane
%head.start.wide = zext i32 %head.start to i64 %n.wide = zext i32 %n to i64 %output.channel = add i64 %head.start.wide, %n.wide
%output.channel.base = mul i64 %output.channel, %length.wide
%output.m.wide = zext i32 %output.m to i64 %output.local = add i64 %output.channel.base, %output.m.wide
%output.index = add i64 %output.row, %output.local
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %previous, i64 %output.index
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
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, RECIPE_STATE %index.base ) #3 { entry:
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%width.double = call double @recipe.from.u32(i32 %head.width)
%scale.default = call double @recipe.sqrt(double %width.double)
%attention.zero = call double @recipe.from.u32(i32 0)
%attention.one = call double @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(double %attention.zero, double %epsilon)
%scale = select i1 %unscaled, double %attention.one, double %scale.default
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
%stage.vector.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.vector.input.index
%stage.vector.input.value = load <16 x double>, ptr addrspace(1) %stage.vector.input.ptr, align 8
br label %attention.matrix.stage.vector.store
attention.matrix.stage.vector.delta:
%stage.vector.delta.index = add i64 %stage.delta.base, %stage.position.global
%stage.vector.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i64 %stage.vector.delta.index
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
%stage.scalar.input.index = add i64 %stage.input.base, %stage.position.global
%stage.scalar.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %stage.scalar.input.index
%stage.scalar.input.value = load double, ptr addrspace(1) %stage.scalar.input.ptr, align 8
br label %attention.matrix.stage.scalar.store
attention.matrix.stage.scalar.delta:
%stage.scalar.delta.index = add i64 %stage.delta.base, %stage.position.global
%stage.scalar.delta.ptr = getelementptr inbounds double, ptr addrspace(1) %delta, i64 %stage.scalar.delta.index
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
%d.query.wide = zext i32 %d.query to i64 %d.channel.wide = zext i32 %d.channel to i64 %d.shared.row = mul i64 %d.query.wide, %head.width.global
%d.shared.local = add i64 %d.shared.row, %d.channel.wide
%d.do.index = add i64 %do.base.global, %d.shared.local
%d.do.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i64 0, i64 %d.do.index
%d.do = load double, ptr addrspace(3) %d.do.ptr, align 8
%d.output.channel = add i64 %head.start.global, %d.channel.wide
%d.output.channel.base = mul i64 %d.output.channel, %length.global
%d.output.local = add i64 %d.output.channel.base, %d.query.wide
%d.output.index = add i64 %output.row, %d.output.local
%d.output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %d.output.index
%d.output = load double, ptr addrspace(1) %d.output.ptr, align 8
%d.term = call double @recipe.mul(double %d.do, double %d.output)
%d.sum.next = call double @recipe.add(double %d.sum, double %d.term)
%d.channel.next = add i32 %d.channel, 1
br label %attention.matrix.d.sum.loop
attention.matrix.d.store:
%d.query.wide.store = zext i32 %d.query to i64 %d.index = add i64 %d.base.global, %d.query.wide.store
%d.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i64 0, i64 %d.index
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
br i1 %matrix.output.more, label %attention.matrix.score.store.check, label %attention.matrix.score.store.done
attention.matrix.score.store.check:
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
%matrix.head.job.global = zext i32 %head.job to i64
%matrix.query.global = zext i32 %matrix.query to i64
%matrix.statistics.base = mul i64 %matrix.head.job.global, %length.global
%matrix.statistics.index = add i64 %matrix.statistics.base, %matrix.query.global
%matrix.maximum.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %matrix.statistics.index
%matrix.maximum = load double, ptr addrspace(1) %matrix.maximum.ptr, align 8
%matrix.denominator.index = add i64 %statistics.rows.global, %matrix.statistics.index
%matrix.denominator.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %matrix.denominator.index
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
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 0, i64 %ds.base.global, i64 %k.base.global, i64 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 1, i64 %ds.base.global, i64 %q.base.global, i64 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
call void @attention_matrix_product(ptr addrspace(1) %previous, i32 2, i64 %p.base.global, i64 %do.base.global, i64 %input.row, i32 %from, i32 %head.start, i32 %length, i32 %head.width, double %scale, i32 %lid, i32 %block)
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
i32 %kv.heads, i32 %value.heads, i32 %index.heads, i32 %index.width, i32 %select.block, i1 %gate, double %epsilon,
i32 %index.mode, i32 %index.dims, i1 %index.pooled, RECIPE_STATE %index.base ) #3 { entry:
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%lid = call i32 @recipe.local.id.x()
%group = call i32 @recipe.group.id.x()
%block = call i32 @recipe.workgroup.size.x()
%groups = udiv i32 %threads, %block
%length = udiv i32 %from, %channels
%head.width = udiv i32 %channels, %heads
%head.width.double = call double @recipe.from.u32(i32 %head.width)
%scale.default = call double @recipe.sqrt(double %head.width.double)
%attention.zero = call double @recipe.from.u32(i32 0)
%attention.one = call double @recipe.from.u32(i32 1)
%unscaled = call i1 @recipe.ogt(double %attention.zero, double %epsilon)
%scale = select i1 %unscaled, double %attention.one, double %scale.default
%scale.wide = call RECIPE_STATE @recipe.decode(double %scale)
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
%dq.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %dq.input.index.wide
%dq.query.value.model = load double, ptr addrspace(1) %dq.input.ptr, align 8
%dq.query.value = call RECIPE_STATE @recipe.decode(double %dq.query.value.model)
%dq.query.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.query.p
store RECIPE_STATE %dq.query.value, ptr addrspace(3) %dq.query.shared.ptr, align RECIPE_STATE_ALIGN
%dq.delta.index = add i32 %dq.output.row, %dq.local %dq.delta.index.wide = add i64 %dq.output.row.wide, %dq.local.wide
%dq.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %dq.delta.index.wide
%dq.delta.value = load RECIPE_STATE, ptr addrspace(1) %dq.delta.ptr, align RECIPE_STATE_ALIGN
%dq.delta.shared.index = add i32 %dq.delta.base.shared, %dq.query.p
%dq.delta.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.delta.shared.index
store RECIPE_STATE %dq.delta.value, ptr addrspace(3) %dq.delta.shared.ptr, align RECIPE_STATE_ALIGN
%dq.gradient.shared.index = add i32 %dq.gradient.base.shared, %dq.query.p
%dq.gradient.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.gradient.shared.index
store RECIPE_STATE %state.zero, ptr addrspace(3) %dq.gradient.shared.ptr, align RECIPE_STATE_ALIGN
%dq.query.p.next = add i32 %dq.query.p, %block
br label %dq.query.stage.loop
dq.query.stage.done:
call void @recipe.local.barrier()
call void @attention_tile_products(ptr addrspace(1) %output, i64 %dq.output.row.wide, i32 %dq.delta.base.shared,
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
%dq.gate.query = udiv i32 %dq.gate.p, %head.width %dq.gate.query.wide = zext i32 %dq.gate.query to i64
%dq.gate.channel = urem i32 %dq.gate.p, %head.width %dq.gate.channel.wide = zext i32 %dq.gate.channel to i64
%dq.gate.position = add i32 %dq.query.base, %dq.gate.query %dq.gate.position.wide = add i64 %dq.query.base.wide, %dq.gate.query.wide
%dq.gate.output.channel = add i32 %dq.head.start, %dq.gate.channel %dq.gate.output.channel.wide = add i64 %dq.head.start.wide, %dq.gate.channel.wide
%dq.gate.channel.base = mul i32 %dq.gate.output.channel, %length %dq.gate.channel.base.wide = mul i64 %dq.gate.output.channel.wide, %attn.length.wide
%dq.gate.local = add i32 %dq.gate.channel.base, %dq.gate.position %dq.gate.local.wide = add i64 %dq.gate.channel.base.wide, %dq.gate.position.wide
%dq.gate.row = add i32 %dq.input.row, %gate.base %dq.gate.row.wide = add i64 %dq.input.row.wide, %attn.gate.base.wide
%dq.gate.index = add i32 %dq.gate.row, %dq.gate.local %dq.gate.index.wide = add i64 %dq.gate.row.wide, %dq.gate.local.wide
%dq.gate.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i64 %dq.gate.index.wide
%dq.gate.value = load double, ptr addrspace(1) %dq.gate.ptr, align 8
%dq.gate.factor.model = call double @recipe.sigmoid(double %dq.gate.value)
%dq.gate.factor = call RECIPE_STATE @recipe.decode(double %dq.gate.factor.model)
%dq.gate.shared.index = add i32 %dq.delta.base.shared, %dq.gate.p
%dq.gate.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.gate.shared.index
%dq.gate.delta = load RECIPE_STATE, ptr addrspace(3) %dq.gate.shared.ptr, align RECIPE_STATE_ALIGN
%dq.gate.output.index = add i32 %dq.output.row, %dq.gate.local %dq.gate.output.index.wide = add i64 %dq.output.row.wide, %dq.gate.local.wide
%dq.gate.output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %dq.gate.output.index.wide
%dq.gate.output.value.model = load double, ptr addrspace(1) %dq.gate.output.ptr, align 8
%dq.gate.output.value = call RECIPE_STATE @recipe.decode(double %dq.gate.output.value.model)
%dq.gate.one = call RECIPE_STATE @recipe.state.from.u1(i1 true)
%dq.gate.complement = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %dq.gate.one, RECIPE_STATE %dq.gate.factor)
%dq.gate.product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dq.gate.delta, RECIPE_STATE %dq.gate.output.value)
%dq.gate.gradient = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dq.gate.product, RECIPE_STATE %dq.gate.complement)
%dq.gate.previous.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %previous, i64 %dq.gate.index.wide
store RECIPE_STATE %dq.gate.gradient, ptr addrspace(1) %dq.gate.previous.ptr, align RECIPE_STATE_ALIGN
%dq.gate.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dq.gate.delta, RECIPE_STATE %dq.gate.factor)
store RECIPE_STATE %dq.gate.scaled, ptr addrspace(3) %dq.gate.shared.ptr, align RECIPE_STATE_ALIGN
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
%dq.key.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dq.key.input.index
%dq.key.value.model = load double, ptr addrspace(1) %dq.key.input.ptr, align 8
%dq.key.value = call RECIPE_STATE @recipe.decode(double %dq.key.value.model)
%dq.key.shared.index = add i32 %dq.key.base.shared, %dq.key.p
%dq.key.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.key.shared.index
store RECIPE_STATE %dq.key.value, ptr addrspace(3) %dq.key.shared.ptr, align RECIPE_STATE_ALIGN
%dq.value.row = add i32 %dq.input.row, %value.plane.base
%dq.value.channel = add i32 %dq.value.head.start, %dq.key.channel.local
%dq.value.channel.base = mul i32 %dq.value.channel, %length
%dq.value.input.local = add i32 %dq.value.channel.base, %dq.key.position
%dq.value.input.index = add i32 %dq.value.row, %dq.value.input.local
%dq.value.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dq.value.input.index
%dq.value.value.model = load double, ptr addrspace(1) %dq.value.input.ptr, align 8
%dq.value.value = call RECIPE_STATE @recipe.decode(double %dq.value.value.model)
%dq.value.shared.index = add i32 %dq.value.base.shared, %dq.key.p
%dq.value.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.value.shared.index
store RECIPE_STATE %dq.value.value, ptr addrspace(3) %dq.value.shared.ptr, align RECIPE_STATE_ALIGN
%dq.key.p.next = add i32 %dq.key.p, %block
br label %dq.key.stage.loop
dq.key.stage.done:
call void @recipe.local.barrier()
br label %dq.key.norm.done
dq.key.norm.done:
%dq.head.job.arg = zext i32 %dq.head.job to i64
%dq.statistics.denominator.base.arg = zext i32 %statistics.denominator.base to i64
call void @attention_tile_derivatives(ptr addrspace(1) %context, i32 0, i32 %dq.key.base.shared,
i32 %dq.delta.base.shared, i32 %dq.value.base.shared, i32 %dq.probability.base.shared,
i32 %dq.derivative.base.shared, i32 %dq.product.base.shared, i32 %dq.query.base,
i32 %dq.key.tile.base, i32 %dq.query.count, i32 %dq.key.count, i32 %tile.n,
i64 %dq.head.job.arg, i64 %attn.length.wide, i64 %dq.statistics.denominator.base.arg, i32 %head.width,
double %scale, i32 %lid, i32 %block, i64 %dq.score.row.base, i32 %blocks, i32 %select.block, i1 %select)
call void @recipe.local.barrier()
br label %dq.accumulate.loop
dq.accumulate.loop:
%dq.accumulate.p = phi i32 [ %lid, %dq.key.norm.done ], [ %dq.accumulate.p.next, %dq.accumulate.store ]
%dq.accumulate.p.more = icmp ult i32 %dq.accumulate.p, %dq.active.query.values
br i1 %dq.accumulate.p.more, label %dq.accumulate.prepare, label %dq.accumulate.done
dq.accumulate.prepare:
%dq.accumulate.query = udiv i32 %dq.accumulate.p, %head.width
%dq.accumulate.channel = urem i32 %dq.accumulate.p, %head.width
%dq.accumulate.gradient.index = add i32 %dq.gradient.base.shared, %dq.accumulate.p
%dq.accumulate.gradient.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.accumulate.gradient.index
%dq.accumulate.initial = load RECIPE_STATE, ptr addrspace(3) %dq.accumulate.gradient.ptr, align RECIPE_STATE_ALIGN
br label %dq.accumulate.key.loop
dq.accumulate.key.loop:
%dq.accumulate.key = phi i32 [ 0, %dq.accumulate.prepare ], [ %dq.accumulate.key.next, %dq.accumulate.key.step ]
%dq.accumulate.value = phi RECIPE_STATE [ %dq.accumulate.initial, %dq.accumulate.prepare ], [ %dq.accumulate.next, %dq.accumulate.key.step ]
%dq.accumulate.key.more = icmp ult i32 %dq.accumulate.key, %dq.key.count
br i1 %dq.accumulate.key.more, label %dq.accumulate.key.step, label %dq.accumulate.store
dq.accumulate.key.step:
%dq.accumulate.pair.row = mul i32 %dq.accumulate.query, %tile.n
%dq.accumulate.pair.local = add i32 %dq.accumulate.pair.row, %dq.accumulate.key
%dq.accumulate.pair.index = add i32 %dq.derivative.base.shared, %dq.accumulate.pair.local
%dq.accumulate.pair.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.accumulate.pair.index
%dq.accumulate.ds = load RECIPE_STATE, ptr addrspace(3) %dq.accumulate.pair.ptr, align RECIPE_STATE_ALIGN
%dq.accumulate.key.row = mul i32 %dq.accumulate.key, %head.width
%dq.accumulate.key.local = add i32 %dq.accumulate.key.row, %dq.accumulate.channel
%dq.accumulate.key.index = add i32 %dq.key.base.shared, %dq.accumulate.key.local
%dq.accumulate.key.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.accumulate.key.index
%dq.accumulate.key.value = load RECIPE_STATE, ptr addrspace(3) %dq.accumulate.key.ptr, align RECIPE_STATE_ALIGN
%dq.accumulate.raw = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dq.accumulate.ds, RECIPE_STATE %dq.accumulate.key.value)
%dq.accumulate.term = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %dq.accumulate.raw, RECIPE_STATE %scale.wide)
%dq.accumulate.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %dq.accumulate.value, RECIPE_STATE %dq.accumulate.term)
%dq.accumulate.key.next = add i32 %dq.accumulate.key, 1
br label %dq.accumulate.key.loop
dq.accumulate.store:
store RECIPE_STATE %dq.accumulate.value, ptr addrspace(3) %dq.accumulate.gradient.ptr, align RECIPE_STATE_ALIGN
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
%dq.store.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %previous, i32 %dq.store.index
%dq.store.shared.index = add i32 %dq.gradient.base.shared, %dq.store.p
%dq.store.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dq.store.shared.index
%dq.store.value = load RECIPE_STATE, ptr addrspace(3) %dq.store.shared.ptr, align RECIPE_STATE_ALIGN
store RECIPE_STATE %dq.store.value, ptr addrspace(1) %dq.store.ptr, align RECIPE_STATE_ALIGN
%dq.store.p.next = add i32 %dq.store.p, %block
br label %dq.store.loop
dq.store.done:
call void @recipe.local.barrier()
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
%dkv.key.gradient.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.key.gradient.index
store RECIPE_STATE %state.zero, ptr addrspace(3) %dkv.key.gradient.ptr, align RECIPE_STATE_ALIGN
%dkv.value.gradient.index = add i32 %dkv.value.gradient.base.shared, %dkv.zero.p
%dkv.value.gradient.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.value.gradient.index
store RECIPE_STATE %state.zero, ptr addrspace(3) %dkv.value.gradient.ptr, align RECIPE_STATE_ALIGN
%dkv.zero.p.next = add i32 %dkv.zero.p, %block
br label %dkv.zero.loop
dkv.zero.done:
call void @recipe.local.barrier()
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
call void @recipe.local.barrier()
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
%dkv.key.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dkv.key.input.index
%dkv.key.value.model = load double, ptr addrspace(1) %dkv.key.input.ptr, align 8
%dkv.key.value = call RECIPE_STATE @recipe.decode(double %dkv.key.value.model)
%dkv.key.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.key.p
store RECIPE_STATE %dkv.key.value, ptr addrspace(3) %dkv.key.shared.ptr, align RECIPE_STATE_ALIGN
%dkv.value.channel = add i32 %dkv.head.value.start, %dkv.channel.local
%dkv.value.channel.base = mul i32 %dkv.value.channel, %length
%dkv.value.local = add i32 %dkv.value.channel.base, %dkv.key.position
%dkv.value.row = add i32 %dkv.input.row, %value.plane.base
%dkv.value.input.index = add i32 %dkv.value.row, %dkv.value.local
%dkv.value.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dkv.value.input.index
%dkv.value.value.model = load double, ptr addrspace(1) %dkv.value.input.ptr, align 8
%dkv.value.value = call RECIPE_STATE @recipe.decode(double %dkv.value.value.model)
%dkv.value.shared.index = add i32 %dkv.value.base.shared, %dkv.key.p
%dkv.value.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.value.shared.index
store RECIPE_STATE %dkv.value.value, ptr addrspace(3) %dkv.value.shared.ptr, align RECIPE_STATE_ALIGN
%dkv.key.p.next = add i32 %dkv.key.p, %block
br label %dkv.key.stage.loop
dkv.key.stage.done:
call void @recipe.local.barrier()
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
%dkv.query.input.ptr = getelementptr inbounds double, ptr addrspace(1) %input, i32 %dkv.query.input.index
%dkv.query.value.model = load double, ptr addrspace(1) %dkv.query.input.ptr, align 8
%dkv.query.value = call RECIPE_STATE @recipe.decode(double %dkv.query.value.model)
%dkv.query.shared.index = add i32 %dkv.query.base.shared, %dkv.query.p
%dkv.query.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.query.shared.index
store RECIPE_STATE %dkv.query.value, ptr addrspace(3) %dkv.query.shared.ptr, align RECIPE_STATE_ALIGN
%dkv.delta.input.index = add i32 %dkv.output.row, %dkv.query.input.local
%dkv.delta.input.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i32 %dkv.delta.input.index
%dkv.delta.value = load RECIPE_STATE, ptr addrspace(1) %dkv.delta.input.ptr, align RECIPE_STATE_ALIGN
%dkv.delta.shared.index = add i32 %dkv.delta.base.shared, %dkv.query.p
%dkv.delta.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.delta.shared.index
store RECIPE_STATE %dkv.delta.value, ptr addrspace(3) %dkv.delta.shared.ptr, align RECIPE_STATE_ALIGN
%dkv.query.p.next = add i32 %dkv.query.p, %block
br label %dkv.query.stage.loop
dkv.query.stage.done:
call void @recipe.local.barrier()
%dkv.output.row.wide.arg = zext i32 %dkv.output.row to i64
call void @attention_tile_products(ptr addrspace(1) %output, i64 %dkv.output.row.wide.arg, i32 %dkv.delta.base.shared,
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
%dkv.gate.factor.model = call double @recipe.sigmoid(double %dkv.gate.value)
%dkv.gate.factor = call RECIPE_STATE @recipe.decode(double %dkv.gate.factor.model)
%dkv.gate.shared.index = add i32 %dkv.delta.base.shared, %dkv.gate.p
%dkv.gate.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.gate.shared.index
%dkv.gate.delta = load RECIPE_STATE, ptr addrspace(3) %dkv.gate.shared.ptr, align RECIPE_STATE_ALIGN
%dkv.gate.scaled = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dkv.gate.delta, RECIPE_STATE %dkv.gate.factor)
store RECIPE_STATE %dkv.gate.scaled, ptr addrspace(3) %dkv.gate.shared.ptr, align RECIPE_STATE_ALIGN
%dkv.gate.p.next = add i32 %dkv.gate.p, %block
br label %dkv.gate.loop
dkv.gate.exit:
call void @recipe.local.barrier()
br label %dkv.gate.done
dkv.gate.done:
%dkv.head.job.arg = zext i32 %dkv.head.job to i64
%dkv.statistics.denominator.base.arg = zext i32 %statistics.denominator.base to i64
call void @attention_tile_derivatives(ptr addrspace(1) %context, i32 %dkv.query.base.shared, i32 0,
i32 %dkv.delta.base.shared, i32 %dkv.value.base.shared, i32 %dkv.probability.base.shared,
i32 %dkv.derivative.base.shared, i32 %dkv.product.base.shared, i32 %dkv.query.base,
i32 %dkv.key.base, i32 %dkv.query.count, i32 %dkv.key.count, i32 %tile.n,
i64 %dkv.head.job.arg, i64 %attn.length.wide, i64 %dkv.statistics.denominator.base.arg, i32 %head.width,
double %scale, i32 %lid, i32 %block, i64 %dkv.score.row.base, i32 %blocks, i32 %select.block, i1 %select)
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
%dkv.accumulate.key.gradient.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.accumulate.key.gradient.index
%dkv.accumulate.key.initial = load RECIPE_STATE, ptr addrspace(3) %dkv.accumulate.key.gradient.ptr, align RECIPE_STATE_ALIGN
%dkv.accumulate.value.gradient.index = add i32 %dkv.value.gradient.base.shared, %dkv.accumulate.p
%dkv.accumulate.value.gradient.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.accumulate.value.gradient.index
%dkv.accumulate.value.initial = load RECIPE_STATE, ptr addrspace(3) %dkv.accumulate.value.gradient.ptr, align RECIPE_STATE_ALIGN
br label %dkv.accumulate.query.loop
dkv.accumulate.query.loop:
%dkv.accumulate.query = phi i32 [ 0, %dkv.accumulate.prepare ], [ %dkv.accumulate.query.next, %dkv.accumulate.query.step ]
%dkv.accumulate.key.value = phi RECIPE_STATE [ %dkv.accumulate.key.initial, %dkv.accumulate.prepare ], [ %dkv.accumulate.key.next, %dkv.accumulate.query.step ]
%dkv.accumulate.value.value = phi RECIPE_STATE [ %dkv.accumulate.value.initial, %dkv.accumulate.prepare ], [ %dkv.accumulate.value.next, %dkv.accumulate.query.step ]
%dkv.accumulate.query.more = icmp ult i32 %dkv.accumulate.query, %dkv.query.count
br i1 %dkv.accumulate.query.more, label %dkv.accumulate.query.step, label %dkv.accumulate.store
dkv.accumulate.query.step:
%dkv.accumulate.pair.row = mul i32 %dkv.accumulate.query, %tile.n
%dkv.accumulate.pair.local = add i32 %dkv.accumulate.pair.row, %dkv.accumulate.key
%dkv.accumulate.probability.index = add i32 %dkv.probability.base.shared, %dkv.accumulate.pair.local
%dkv.accumulate.probability.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.accumulate.probability.index
%dkv.accumulate.probability = load RECIPE_STATE, ptr addrspace(3) %dkv.accumulate.probability.ptr, align RECIPE_STATE_ALIGN
%dkv.accumulate.derivative.index = add i32 %dkv.derivative.base.shared, %dkv.accumulate.pair.local
%dkv.accumulate.derivative.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.accumulate.derivative.index
%dkv.accumulate.derivative = load RECIPE_STATE, ptr addrspace(3) %dkv.accumulate.derivative.ptr, align RECIPE_STATE_ALIGN
%dkv.accumulate.query.row = mul i32 %dkv.accumulate.query, %head.width
%dkv.accumulate.query.local = add i32 %dkv.accumulate.query.row, %dkv.accumulate.channel
%dkv.accumulate.query.index = add i32 %dkv.query.base.shared, %dkv.accumulate.query.local
%dkv.accumulate.query.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.accumulate.query.index
%dkv.accumulate.query.value = load RECIPE_STATE, ptr addrspace(3) %dkv.accumulate.query.ptr, align RECIPE_STATE_ALIGN
%dkv.accumulate.delta.index = add i32 %dkv.delta.base.shared, %dkv.accumulate.query.local
%dkv.accumulate.delta.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.accumulate.delta.index
%dkv.accumulate.delta = load RECIPE_STATE, ptr addrspace(3) %dkv.accumulate.delta.ptr, align RECIPE_STATE_ALIGN
%dkv.accumulate.key.raw = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dkv.accumulate.derivative, RECIPE_STATE %dkv.accumulate.query.value)
%dkv.accumulate.key.term = call RECIPE_STATE @recipe.state.div(RECIPE_STATE %dkv.accumulate.key.raw, RECIPE_STATE %scale.wide)
%dkv.accumulate.key.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %dkv.accumulate.key.value, RECIPE_STATE %dkv.accumulate.key.term)
%dkv.accumulate.value.term = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dkv.accumulate.probability, RECIPE_STATE %dkv.accumulate.delta)
%dkv.accumulate.value.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %dkv.accumulate.value.value, RECIPE_STATE %dkv.accumulate.value.term)
%dkv.accumulate.query.next = add i32 %dkv.accumulate.query, 1
br label %dkv.accumulate.query.loop
dkv.accumulate.store:
store RECIPE_STATE %dkv.accumulate.key.value, ptr addrspace(3) %dkv.accumulate.key.gradient.ptr, align RECIPE_STATE_ALIGN
store RECIPE_STATE %dkv.accumulate.value.value, ptr addrspace(3) %dkv.accumulate.value.gradient.ptr, align RECIPE_STATE_ALIGN
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
%dkv.store.key.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %previous, i32 %dkv.store.key.index
%dkv.store.key.shared.index = add i32 %dkv.key.gradient.base.shared, %dkv.store.p
%dkv.store.key.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.store.key.shared.index
%dkv.store.key.value = load RECIPE_STATE, ptr addrspace(3) %dkv.store.key.shared.ptr, align RECIPE_STATE_ALIGN
store RECIPE_STATE %dkv.store.key.value, ptr addrspace(1) %dkv.store.key.ptr, align RECIPE_STATE_ALIGN
br label %dkv.store.next
dkv.store.value.step:
%dkv.store.value.channel = add i32 %dkv.store.value.head.start, %dkv.store.channel.local
%dkv.store.value.channel.base = mul i32 %dkv.store.value.channel, %length
%dkv.store.value.local = add i32 %dkv.store.value.channel.base, %dkv.store.key
%dkv.store.value.row = add i32 %dkv.input.row, %value.plane.base
%dkv.store.value.index = add i32 %dkv.store.value.row, %dkv.store.value.local
%dkv.store.value.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %previous, i32 %dkv.store.value.index
%dkv.store.value.shared.index = add i32 %dkv.value.gradient.base.shared, %dkv.store.p
%dkv.store.value.shared.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %dkv.store.value.shared.index
%dkv.store.value.value = load RECIPE_STATE, ptr addrspace(3) %dkv.store.value.shared.ptr, align RECIPE_STATE_ALIGN
store RECIPE_STATE %dkv.store.value.value, ptr addrspace(1) %dkv.store.value.ptr, align RECIPE_STATE_ALIGN
br label %dkv.store.next
dkv.store.next:
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
	ptr addrspace(1) %context, i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels, i32 %time.begin, i32 %time.span, i32 %gates,
	i1 %has.bias, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads, i64 %weight.base, i32 %decode, i1 %coded, i32 %cell.activation ) #3 { entry: %tid = call i32 @llvm.amdgcn.workitem.id.x()
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
%precompute.dense = getelementptr double, ptr addrspace(1) %weights, i64 %precompute.weight.offset
%precompute.weights = select i1 %weight.packed, ptr addrspace(1) %weights, ptr addrspace(1) %precompute.dense
%precompute.base = add i64 %weight.base, %precompute.weight.offset
%precompute.context.offset = mul i64 %precompute.gate.wide, %gate.batch.wide
%precompute.context = getelementptr inbounds double, ptr addrspace(1) %context, i64 %precompute.context.offset
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
%input.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %input.index
%input.sum = load double, ptr addrspace(1) %input.ptr, align 8 br label %state.sum.loop state.sum.loop:
%state.channel = phi i32 [ 0, %input.load ], [ %state.next, %state.weight.ready ]
%state.sum = phi double [ %input.sum, %input.load ], [ %state.sum.next, %state.weight.ready ]
%state.more = icmp ult i32 %state.channel, %out.channels br i1 %state.more, label %state.sum.step, label %gate.activate
state.sum.step: %previous.time = sub i32 %time, 1 %previous.safe = select i1 %previous.exists, i32 %previous.time, i32 0 %state.channel.wide = zext i32 %state.channel to i64 %previous.safe.wide = zext i32 %previous.safe to i64
%state.channel.base = mul i64 %state.channel.wide, %length.wide %previous.local = add i64 %state.channel.base, %previous.safe.wide
%previous.index = add i64 %output.row.base, %previous.local
%previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %previous.index
%previous.loaded = load double, ptr addrspace(1) %previous.ptr, align 8
%previous = select i1 %previous.exists, double %previous.loaded, double 0.0 %candidate.gate = icmp eq i32 %gate, 2
%gru = icmp eq i32 %gates, 3 %reset.candidate = and i1 %gru, %candidate.gate
%reset.channel.base = mul i64 %state.channel.wide, %length.wide %reset.local = add i64 %reset.channel.base, %time.wide
%reset.row.index = add i64 %output.row.base, %reset.local %reset.base = add i64 %gate.batch.wide, %reset.row.index
%reset.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %reset.base
%reset = load double, ptr addrspace(1) %reset.ptr, align 8 %reset.state = call double @recipe.mul(double %reset, double %previous)
%state.value = select i1 %reset.candidate, double %reset.state, double %previous
%state.weight.base = add i64 %gate.weight.base, %input.matrix.wide %state.weight.row = mul i64 %state.channel.wide, %out.channels.wide
%hidden.wide.state = zext i32 %hidden to i64 %state.weight.local = add i64 %state.weight.row, %hidden.wide.state
%state.weight.index = add i64 %state.weight.base, %state.weight.local
br i1 %weight.packed, label %state.weight.packed, label %state.weight.direct
state.weight.direct:
%state.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %state.weight.index
%state.weight.loaded = load double, ptr addrspace(1) %state.weight.ptr, align 8
br label %state.weight.ready
state.weight.packed:
%state.weight.decode.index = add i64 %weight.base.wide, %state.weight.index
%state.weight.decoded = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %state.weight.decode.index, i32 %decode)
br label %state.weight.ready
state.weight.ready:
%state.weight = phi double [ %state.weight.loaded, %state.weight.direct ], [ %state.weight.decoded, %state.weight.packed ]
%state.product = call double @recipe.mul(double %state.value, double %state.weight) %state.sum.next = call double @recipe.add(double %state.sum, double %state.product)
%state.next = add nuw i32 %state.channel, 1 br label %state.sum.loop gate.activate:
	%bias.base = add i64 %gate.weight.base, %matrix.span.wide %bias.hidden = zext i32 %hidden to i64 %bias.index = add i64 %bias.base, %bias.hidden
br i1 %weight.packed, label %gate.bias.packed, label %gate.bias.direct
gate.bias.direct:
%bias.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %bias.index
%bias.loaded = load double, ptr addrspace(1) %bias.ptr, align 8
br label %gate.bias.ready
gate.bias.packed:
%bias.decode.index = add i64 %weight.base.wide, %bias.index
%bias.decoded = call double @recipe.model.decode(ptr addrspace(1) %weights, i64 %bias.decode.index, i32 %decode)
br label %gate.bias.ready
gate.bias.ready:
	%bias = phi double [ %bias.loaded, %gate.bias.direct ], [ %bias.decoded, %gate.bias.packed ]
	%bias.value = select i1 %has.bias, double %bias, double 0.0
	%linear = call double @recipe.add(double %state.sum, double %bias.value)
%rnn = icmp eq i32 %gates, 1 %last.gate = sub i32 %gates, 1 %candidate = icmp eq i32 %gate, %last.gate
%use.tanh = or i1 %rnn, %candidate %tanh.value = call double @recipe.tanh(double %linear)
%sigmoid.value = call double @sigmoid(double %linear)
%builtin.value = select i1 %use.tanh, double %tanh.value, double %sigmoid.value
%coded.relu = icmp eq i32 %cell.activation, 1 %coded.tanh = icmp eq i32 %cell.activation, 2 %coded.sigmoid = icmp eq i32 %cell.activation, 3
%coded.positive = call i1 @recipe.ogt(double %linear, double 0.0) %coded.relu.value = select i1 %coded.positive, double %linear, double 0.0
%coded.a = select i1 %coded.relu, double %coded.relu.value, double %linear
%coded.b = select i1 %coded.tanh, double %tanh.value, double %coded.a
%coded.value = select i1 %coded.sigmoid, double %sigmoid.value, double %coded.b
%gate.value = select i1 %coded, double %coded.value, double %builtin.value br label %gate.store gate.store:
%gate.context.base = mul i64 %gate.wide, %gate.batch.wide %gate.hidden.base = mul i64 %hidden.wide, %length.wide
%gate.local = add i64 %gate.hidden.base, %time.wide %gate.row.local = add i64 %output.row.base, %gate.local
%gate.index = add i64 %gate.context.base, %gate.row.local
%gate.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %gate.index
store double %gate.value, ptr addrspace(1) %gate.ptr, align 8 %hidden.next = add nuw i32 %hidden, 1
br label %hidden.loop hidden.done: %gate.next = add nuw i32 %gate, 1 br label %gate.loop output.loop:
%output.hidden = phi i32 [ 0, %gate.loop ], [ %output.next, %output.store ]
%output.more = icmp ult i32 %output.hidden, %out.channels br i1 %output.more, label %output.step, label %output.done
output.step: %output.hidden.wide = zext i32 %output.hidden to i64 %output.hidden.base = mul i64 %output.hidden.wide, %length.wide %output.local = add i64 %output.hidden.base, %time.wide
%output.index = add i64 %output.row.base, %output.local
%gate0.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %output.index
%gate0 = load double, ptr addrspace(1) %gate0.ptr, align 8
%is.gru = icmp eq i32 %gates, 3 %is.lstm = icmp eq i32 %gates, 4
%gate1.raw = add i64 %gate.batch.wide, %output.index %gate1.index = select i1 %is.lstm, i64 %gate1.raw, i64 %output.index
%gate1.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %gate1.index
%gate1 = load double, ptr addrspace(1) %gate1.ptr, align 8 %gate2.base = mul i64 %gate.batch.wide, 2
%gate2.raw = add i64 %gate2.base, %output.index %gate2.valid = or i1 %is.gru, %is.lstm
%gate2.index = select i1 %gate2.valid, i64 %gate2.raw, i64 %output.index
%gate2.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %gate2.index
%gate2 = load double, ptr addrspace(1) %gate2.ptr, align 8 %gate3.base = mul i64 %gate.batch.wide, 3
%gate3.raw = add i64 %gate3.base, %output.index %gate3.index = select i1 %is.lstm, i64 %gate3.raw, i64 %output.index
%gate3.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %gate3.index
%gate3 = load double, ptr addrspace(1) %gate3.ptr, align 8 %output.previous.time = sub i32 %time, 1
%output.previous.safe = select i1 %previous.exists, i32 %output.previous.time, i32 0 %output.previous.safe.wide = zext i32 %output.previous.safe to i64
%output.previous.local = add i64 %output.hidden.base, %output.previous.safe.wide
%output.previous.index = add i64 %output.row.base, %output.previous.local
%output.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.previous.index
%output.previous.loaded = load double, ptr addrspace(1) %output.previous.ptr, align 8
%output.previous = select i1 %previous.exists, double %output.previous.loaded, double 0.0
%one.update = call double @recipe.sub(double 1.0, double %gate0) %gru.old = call double @recipe.mul(double %gate0, double %output.previous)
%gru.new = call double @recipe.mul(double %one.update, double %gate2) %gru.value = call double @recipe.add(double %gru.old, double %gru.new)
%gates.wide = zext i32 %gates to i64 %cell.base = mul i64 %gate.batch.wide, %gates.wide %cell.index = add i64 %cell.base, %output.index
%cell.previous.index = add i64 %cell.base, %output.previous.index
%cell.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %cell.previous.index
%cell.previous.loaded = load double, ptr addrspace(1) %cell.previous.ptr, align 8
%cell.previous = select i1 %previous.exists, double %cell.previous.loaded, double 0.0
%cell.old = call double @recipe.mul(double %gate1, double %cell.previous) %cell.new = call double @recipe.mul(double %gate0, double %gate3)
%cell = call double @recipe.add(double %cell.old, double %cell.new)
%cell.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %cell.index
store double %cell, ptr addrspace(1) %cell.ptr, align 8 %cell.tanh = call double @recipe.tanh(double %cell)
%lstm.value = call double @recipe.mul(double %gate2, double %cell.tanh)
%rnn.or.gru = select i1 %is.gru, double %gru.value, double %gate0
%output.value = select i1 %is.lstm, double %lstm.value, double %rnn.or.gru br label %output.store output.store:
%output.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %output.index
store double %output.value, ptr addrspace(1) %output.ptr, align 8 %output.next = add nuw i32 %output.hidden, 1
br label %output.loop output.done: %time.next = add nuw i32 %time, 1 br label %time.loop time.done:
%row.next = add i32 %row, %threads br label %row.loop exit: ret void }
define internal void @contraction_reverse_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %delta, ptr addrspace(1) %previous, ptr addrspace(1) %gradient, i1 %write.input, i1 %has.bias, i1 %relu, i1 %matrix.gradient,
i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %kernel, i32 %offset,
i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) RECIPE_CONTRACTION_BODY { entry:
%sums = alloca [RECIPE_REGISTER_COUNT x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5)
%bias.sums = alloca [RECIPE_REGISTER_N x RECIPE_STATE], align RECIPE_STATE_ALIGN, addrspace(5)
%state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %lid = call i32 @recipe.local.id.x() %group = call i32 @recipe.group.id.x() %block = call i32 @recipe.workgroup.size.x() %groups = udiv i32 %threads, %block
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
%gradient.scratch = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gradient, i32 RECIPE_GRADIENT_SCRATCH_BASE
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
%gradient.a.row = udiv i32 %gradient.a.global, %out.length %gradient.a.position = urem i32 %gradient.a.global, %out.length %gradient.a.row.wide = zext i32 %gradient.a.row to i64 %gradient.a.position.wide = zext i32 %gradient.a.position to i64 %gradient.a.row.base = mul i64 %gradient.a.row.wide, %in.elements.wide %gradient.a.term = add i32 %gradient.m.base, %gradient.a.m %gradient.a.term.wide = zext i32 %gradient.a.term to i64
%gradient.a.tile.index = call i32 @contraction_vector_a_index(i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br i1 %gradient.a.vector, label %gradient.load.a.vector, label %gradient.load.a.scalar
gradient.load.a.vector:
%gradient.a.vector.index = add i64 %gradient.a.row.base, %gradient.a.term.wide
%gradient.a.vector.source = getelementptr inbounds double, ptr addrspace(1) %input, i64 %gradient.a.vector.index
%gradient.a.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %gradient.a.vector.source, align 8
call void @contraction_stage_a_columns(<RECIPE_FRAGMENT_K x double> %gradient.a.vector.value, i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.a.scalar:
%gradient.a.value = call double @contraction_input( ptr addrspace(1) %input, i64 %gradient.a.row.base, i32 %gradient.a.position, i32 %gradient.a.term, i32 %span, i32 %in.length, i1 %is.conv )
br label %gradient.load.store
gradient.load.b.step: %gradient.b.local = sub i32 %gradient.load, %gradient.a.count %gradient.b.r = udiv i32 %gradient.b.local, %gradient.b.columns %gradient.b.column = urem i32 %gradient.b.local, %gradient.b.columns %gradient.b.n = mul i32 %gradient.b.column, %gradient.b.width %gradient.b.global = add i32 %gradient.r.base, %gradient.b.r
%gradient.b.row = udiv i32 %gradient.b.global, %out.length %gradient.b.position = urem i32 %gradient.b.global, %out.length %gradient.b.filter = add i32 %gradient.n.base, %gradient.b.n
%gradient.b.row.wide = zext i32 %gradient.b.row to i64 %gradient.b.position.wide = zext i32 %gradient.b.position to i64 %gradient.b.filter.wide = zext i32 %gradient.b.filter to i64 %gradient.b.row.base = mul i64 %gradient.b.row.wide, %out.elements.wide %gradient.b.filter.base = mul i64 %gradient.b.filter.wide, %out.length.wide %gradient.b.local.index = add i64 %gradient.b.filter.base, %gradient.b.position.wide %gradient.b.index = add i64 %gradient.b.row.base, %gradient.b.local.index
%gradient.b.model.elements = mul i32 %gradient.tile.m, %gradient.tile.k
%gradient.b.tile.base = call i32 @contraction_state_after_model(i32 %gradient.b.model.elements)
%gradient.b.tile.local = call i32 @contraction_vector_b_index(i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.n, i32 %gradient.tile.k) %gradient.b.tile.index = add i32 %gradient.b.tile.base, %gradient.b.tile.local
br i1 %gradient.b.vector, label %gradient.load.b.vector, label %gradient.load.b.scalar
gradient.load.b.vector:
%gradient.b.vector.value = call <RECIPE_FRAGMENT_K x RECIPE_STATE> @contraction_delta_vector16_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %gradient.b.index, i1 %relu)
call void @contraction_stage_b_fragment_state(<RECIPE_FRAGMENT_K x RECIPE_STATE> %gradient.b.vector.value, i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.b.scalar:
%gradient.b.value = call RECIPE_STATE @contraction_delta_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %gradient.b.index, i1 %relu)
%gradient.load.b.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %gradient.b.tile.index store RECIPE_STATE %gradient.b.value, ptr addrspace(3) %gradient.load.b.ptr, align RECIPE_STATE_ALIGN
br label %gradient.load.advance
gradient.load.store:
%gradient.load.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %gradient.a.tile.index store double %gradient.a.value, ptr addrspace(3) %gradient.load.ptr, align 8
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
call void @recipe.local.barrier()
br label %gradient.scalar.ready
gradient.scalar.ready:
call void @contraction_bias_accumulate_state(ptr addrspace(5) %bias.sums, ptr addrspace(1) %gradient.destination, i1 %gradient.bias.enable, i1 %gradient.r.first.tile, i1 %gradient.r.last.tile, i32 %lid, i32 %block, i32 %gradient.n.base, i32 %gradient.n.count, i32 %gradient.r.count, i32 %out.channels, i32 %window, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %gradient.store.offset)
call void @contraction_vector_accumulate_bs(ptr addrspace(5) %sums, i1 %gradient.lane.active, i1 %gradient.method.store, i32 %lid, i32 %gradient.lane.k, i32 %gradient.k.lanes, i32 %gradient.output.lane, i32 %gradient.output.lanes, i32 %gradient.output.m.base, i32 %gradient.output.n.base, i32 %gradient.m.count, i32 %gradient.n.count, i32 %gradient.r.count, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.product.done
gradient.product.done:
call void @recipe.local.barrier()
br i1 %gradient.r.more, label %gradient.tile.done, label %gradient.store.loop gradient.tile.done: br label %gradient.tile.loop
gradient.store.loop:
%gradient.store.register = phi i32 [ 0, %gradient.product.done ], [ %gradient.store.register.next, %gradient.store.next ] %gradient.store.more = icmp ult i32 %gradient.store.register, RECIPE_REGISTER_COUNT br i1 %gradient.store.more, label %gradient.store.check, label %gradient.bias.store.check
gradient.store.check: %gradient.store.register.m = urem i32 %gradient.store.register, RECIPE_REGISTER_M %gradient.store.register.n = udiv i32 %gradient.store.register, RECIPE_REGISTER_M %gradient.store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.register.valid = call i1 @contraction_output_register_valid(i32 %gradient.store.register)
%gradient.store.output.m.valid = icmp ult i32 %gradient.store.output.m.raw, %gradient.m.count %gradient.store.output.n.valid = icmp ult i32 %gradient.store.output.n.raw, %gradient.n.count %gradient.store.output.valid = and i1 %gradient.store.output.m.valid, %gradient.store.output.n.valid %gradient.store.lane.active = and i1 %gradient.method.store, %gradient.store.output.valid %gradient.store.active = and i1 %gradient.store.lane.active, %gradient.store.register.valid br i1 %gradient.store.active, label %gradient.store, label %gradient.store.next
gradient.store: %gradient.store.filter = add i32 %gradient.n.base, %gradient.store.output.n.raw %gradient.store.term = add i32 %gradient.m.base, %gradient.store.output.m.raw %gradient.store.filter.wide = zext i32 %gradient.store.filter to i64 %gradient.store.term.wide = zext i32 %gradient.store.term to i64 %gradient.store.filter.base = mul i64 %gradient.store.filter.wide, %window.wide %gradient.store.local = add i64 %gradient.store.filter.base, %gradient.store.term.wide %gradient.store.index = add i64 %gradient.store.offset.wide, %gradient.store.local
%gradient.store.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %gradient.destination, i64 %gradient.store.index %gradient.store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %gradient.store.register %gradient.store.sum.wide = load RECIPE_STATE, ptr addrspace(5) %gradient.store.sum.ptr, align RECIPE_STATE_ALIGN store RECIPE_STATE %gradient.store.sum.wide, ptr addrspace(1) %gradient.store.ptr, align RECIPE_STATE_ALIGN
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
%previous.method.store = call i1 @contraction_vector_store_lane(i1 %previous.lane.store, i32 %lid)
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
previous.load.a.step: %previous.a.m = udiv i32 %previous.load, %previous.a.columns %previous.a.column = urem i32 %previous.load, %previous.a.columns %previous.a.r = mul i32 %previous.a.column, %previous.a.width %previous.a.term = add i32 %previous.r.base, %previous.a.r %previous.a.term.wide = zext i32 %previous.a.term to i64
%previous.a.filter = udiv i32 %previous.a.term, %span %previous.a.kernel = urem i32 %previous.a.term, %span %previous.a.global = add i32 %previous.m.base, %previous.a.m %previous.a.row = udiv i32 %previous.a.global, %in.length %previous.a.position = urem i32 %previous.a.global, %in.length
%previous.a.low = icmp uge i32 %previous.a.position, %previous.a.kernel %previous.a.position.raw = sub i32 %previous.a.position, %previous.a.kernel %previous.a.high = icmp ult i32 %previous.a.position.raw, %out.length %previous.a.valid = and i1 %previous.a.low, %previous.a.high
%previous.a.position.safe = select i1 %previous.a.valid, i32 %previous.a.position.raw, i32 0 %previous.a.row.wide = zext i32 %previous.a.row to i64 %previous.a.filter.wide = zext i32 %previous.a.filter to i64 %previous.a.position.safe.wide = zext i32 %previous.a.position.safe to i64 %previous.a.row.base = mul i64 %previous.a.row.wide, %out.elements.wide %previous.a.filter.base = mul i64 %previous.a.filter.wide, %out.length.wide
%previous.a.local = add i64 %previous.a.filter.base, %previous.a.position.safe.wide %previous.a.index = add i64 %previous.a.row.base, %previous.a.local %previous.a.tile.index = call i32 @contraction_vector_a_index(i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br i1 %previous.a.vector, label %previous.load.a.vector, label %previous.load.a.scalar
previous.load.a.vector:
%previous.a.vector.delta = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %previous.a.index
%previous.a.vector.output = getelementptr inbounds double, ptr addrspace(1) %output, i64 %previous.a.index
%previous.a.vector.delta.value = load <RECIPE_FRAGMENT_K x RECIPE_STATE>, ptr addrspace(1) %previous.a.vector.delta, align RECIPE_STATE_ALIGN
%previous.a.vector.output.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %previous.a.vector.output, align 8
call void @contraction_stage_delta_a_fragment_state(<RECIPE_FRAGMENT_K x RECIPE_STATE> %previous.a.vector.delta.value, <RECIPE_FRAGMENT_K x double> %previous.a.vector.output.value, i1 %relu, i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.a.scalar:
%previous.a.raw = call RECIPE_STATE @contraction_delta_state(ptr addrspace(1) %delta, ptr addrspace(1) %output, i64 %previous.a.index, i1 %relu)
%previous.a.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false)
%previous.a.value = select i1 %previous.a.valid, RECIPE_STATE %previous.a.raw, RECIPE_STATE %previous.a.zero
%previous.load.a.ptr = getelementptr RECIPE_STATE, ptr addrspace(3) @contraction_tile, i32 %previous.a.tile.index store RECIPE_STATE %previous.a.value, ptr addrspace(3) %previous.load.a.ptr, align RECIPE_STATE_ALIGN
br label %previous.load.advance
previous.load.b.step: %previous.b.local = sub i32 %previous.load, %previous.a.count %previous.b.r = udiv i32 %previous.b.local, %previous.b.columns %previous.b.column = urem i32 %previous.b.local, %previous.b.columns %previous.b.n = mul i32 %previous.b.column, %previous.b.width %previous.b.term = add i32 %previous.r.base, %previous.b.r
%previous.b.filter = udiv i32 %previous.b.term, %span %previous.b.kernel = urem i32 %previous.b.term, %span %previous.b.channel = add i32 %previous.n.base, %previous.b.n %previous.b.filter.wide = zext i32 %previous.b.filter to i64 %previous.b.channel.wide = zext i32 %previous.b.channel to i64 %previous.b.kernel.wide = zext i32 %previous.b.kernel to i64 %previous.b.filter.base = mul i64 %previous.b.filter.wide, %window.wide
%previous.b.channel.base = mul i64 %previous.b.channel.wide, %previous.span.wide %previous.b.channel.local = add i64 %previous.b.channel.base, %previous.b.kernel.wide %previous.b.index = add i64 %previous.b.filter.base, %previous.b.channel.local
%previous.a.state.elements = mul i32 %previous.tile.m, %previous.tile.k
%previous.b.tile.base = call i32 @contraction_model_after_state(i32 %previous.a.state.elements)
%previous.b.tile.local = call i32 @contraction_vector_b_index(i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.n, i32 %previous.tile.k) %previous.b.tile.index = add i32 %previous.b.tile.base, %previous.b.tile.local
br i1 %previous.b.vector, label %previous.load.b.vector, label %previous.load.b.scalar
previous.load.b.vector:
%previous.b.vector.source = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %previous.b.index
%previous.b.vector.value = load <RECIPE_FRAGMENT_K x double>, ptr addrspace(1) %previous.b.vector.source, align 8
call void @contraction_stage_b_fragment_after_state(<RECIPE_FRAGMENT_K x double> %previous.b.vector.value, i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.b.scalar:
%previous.b.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %previous.b.index
%previous.b.value = load double, ptr addrspace(1) %previous.b.ptr, align 8
br label %previous.load.store
previous.load.store:
%previous.load.ptr = getelementptr [0 x double], ptr addrspace(3) @contraction_tile, i32 0, i32 %previous.b.tile.index store double %previous.b.value, ptr addrspace(3) %previous.load.ptr, align 8
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
call void @recipe.local.barrier()
call void @contraction_vector_accumulate_as(ptr addrspace(5) %sums, i1 %previous.lane.active, i1 %previous.method.store, i32 %lid, i32 %previous.lane.k, i32 %previous.k.lanes, i32 %previous.output.lane, i32 %previous.lanes, i32 %previous.output.m.base, i32 %previous.output.n.base, i32 %previous.m.count, i32 %previous.n.count, i32 %previous.r.count, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k) call void @recipe.local.barrier()
%previous.r.next = add i32 %previous.r.base, %previous.r.count %previous.r.more = icmp ult i32 %previous.r.next, %previous.r.total br i1 %previous.r.more, label %previous.tile.done, label %previous.store.loop previous.tile.done: br label %previous.tile.loop previous.store.loop:
%previous.store.register = phi i32 [ 0, %previous.load.ready ], [ %previous.store.register.next, %previous.store.next ] %previous.store.more = icmp ult i32 %previous.store.register, RECIPE_REGISTER_COUNT br i1 %previous.store.more, label %previous.store.check, label %previous.job.done
previous.store.check: %previous.store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.register.valid = call i1 @contraction_output_register_valid(i32 %previous.store.register)
%previous.store.output.m.valid = icmp ult i32 %previous.store.output.m.raw, %previous.m.count %previous.store.output.n.valid = icmp ult i32 %previous.store.output.n.raw, %previous.n.count %previous.store.output.valid = and i1 %previous.store.output.m.valid, %previous.store.output.n.valid %previous.lane.output.active = and i1 %previous.method.store, %previous.store.output.valid %previous.store.active = and i1 %previous.lane.output.active, %previous.store.register.valid br i1 %previous.store.active, label %previous.store, label %previous.store.next
previous.store: %previous.store.m.global = add i32 %previous.m.base, %previous.store.output.m.raw %previous.store.channel = add i32 %previous.n.base, %previous.store.output.n.raw %previous.store.m.global.wide = zext i32 %previous.store.m.global to i64 %previous.store.channel.wide = zext i32 %previous.store.channel to i64 %previous.store.row.wide = udiv i64 %previous.store.m.global.wide, %in.length.wide %previous.store.position.wide = urem i64 %previous.store.m.global.wide, %in.length.wide
%previous.store.row.base = mul i64 %previous.store.row.wide, %in.elements.wide %previous.store.channel.base = mul i64 %previous.store.channel.wide, %in.length.wide %previous.store.local = add i64 %previous.store.channel.base, %previous.store.position.wide %previous.store.index = add i64 %previous.store.row.base, %previous.store.local %previous.store.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %previous, i64 %previous.store.index
%previous.store.old = load RECIPE_STATE, ptr addrspace(1) %previous.store.ptr, align RECIPE_STATE_ALIGN %previous.store.sum.ptr = getelementptr [RECIPE_REGISTER_COUNT x RECIPE_STATE], ptr addrspace(5) %sums, i32 0, i32 %previous.store.register %previous.store.sum.wide = load RECIPE_STATE, ptr addrspace(5) %previous.store.sum.ptr, align RECIPE_STATE_ALIGN %previous.store.value = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %previous.store.old, RECIPE_STATE %previous.store.sum.wide) store RECIPE_STATE %previous.store.value, ptr addrspace(1) %previous.store.ptr, align RECIPE_STATE_ALIGN br label %previous.store.next
previous.store.next: %previous.store.register.next = add i32 %previous.store.register, 1 br label %previous.store.loop previous.job.done: %previous.job.next = add i32 %previous.job, %groups br label %previous.job.loop exit: ret void }
define internal void @scan_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output,
ptr addrspace(1) %context, ptr addrspace(1) %backward, ptr addrspace(1) %delta, ptr addrspace(1) %previous,
ptr addrspace(1) %gradient, i1 %write.input, i32 %rows, i32 %in.channels,
i32 %length, i32 %out.channels, i32 %gates, i1 %has.bias, i32 %parameters, i32 %offset,
i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads, i1 %coded, i32 %cell.activation ) #3 { entry:
	%tid = call i32 @llvm.amdgcn.workitem.id.x() %state.zero = call RECIPE_STATE @recipe.state.from.u1(i1 false) %state.one = call RECIPE_STATE @recipe.state.from.u1(i1 true) %scan.in.channels.wide = zext i32 %in.channels to i64 %scan.length.wide = zext i32 %length to i64 %scan.out.channels.wide = zext i32 %out.channels to i64 %scan.rows.wide = zext i32 %rows to i64 %scan.gates.wide = zext i32 %gates to i64 %scan.parameters.wide = zext i32 %parameters to i64 %in.elements = mul i32 %in.channels, %length %scan.in.elements.wide = mul i64 %scan.in.channels.wide, %scan.length.wide
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
%clear.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.clear.index.wide
store RECIPE_STATE %state.zero, ptr addrspace(1) %clear.ptr, align RECIPE_STATE_ALIGN %clear.next = add nuw i32 %clear.p, 1
br label %clear.gradient.loop clear.state.loop:
%clear.h = phi i32 [ 0, %clear.gradient.loop ], [ %clear.h.next, %clear.state.step ]
%scratch.base.0 = mul i32 %rows, %parameters %scan.scratch.base.0.wide = mul i64 %scan.rows.wide, %scan.parameters.wide %scratch.base = add i32 %row.gradient.base, %scratch.base.0 %scan.scratch.base.wide = add i64 %scan.row.gradient.base.wide, %scan.scratch.base.0.wide
%scratch.row = mul i32 %row, %out.channels %scan.scratch.row.wide = mul i64 %scan.row.wide, %scan.out.channels.wide %dh.start = add i32 %scratch.base, %scratch.row %scan.dh.start.wide = add i64 %scan.scratch.base.wide, %scan.scratch.row.wide
%dc.base.0 = mul i32 %rows, %out.channels %scan.dc.base.0.wide = mul i64 %scan.rows.wide, %scan.out.channels.wide %dc.base = add i32 %scratch.base, %dc.base.0 %scan.dc.base.wide = add i64 %scan.scratch.base.wide, %scan.dc.base.0.wide
%dc.start = add i32 %dc.base, %scratch.row %scan.dc.start.wide = add i64 %scan.dc.base.wide, %scan.scratch.row.wide %clear.h.more = icmp ult i32 %clear.h, %out.channels
br i1 %clear.h.more, label %clear.state.step, label %time.loop clear.state.step:
%clear.dh.index = add i32 %dh.start, %clear.h %clear.dc.index = add i32 %dc.start, %clear.h %scan.clear.h.wide = zext i32 %clear.h to i64 %scan.clear.dh.index.wide = add i64 %scan.dh.start.wide, %scan.clear.h.wide %scan.clear.dc.index.wide = add i64 %scan.dc.start.wide, %scan.clear.h.wide
%clear.dh.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.clear.dh.index.wide
%clear.dc.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.clear.dc.index.wide
store RECIPE_STATE %state.zero, ptr addrspace(1) %clear.dh.ptr, align RECIPE_STATE_ALIGN store RECIPE_STATE %state.zero, ptr addrspace(1) %clear.dc.ptr, align RECIPE_STATE_ALIGN
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
%rnn.index = add i32 %row.output.base, %rnn.local %scan.rnn.index.wide = add i64 %scan.row.output.base.wide, %scan.rnn.local.wide %rnn.dy.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %scan.rnn.index.wide %rnn.future.index = add i32 %dh.start, %rnn.hidden %scan.rnn.future.index.wide = add i64 %scan.dh.start.wide, %rnn.hidden.wide
%rnn.future.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.rnn.future.index.wide
%rnn.gate.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.rnn.index.wide
%rnn.dy = load RECIPE_STATE, ptr addrspace(1) %rnn.dy.ptr, align RECIPE_STATE_ALIGN
%rnn.future = load RECIPE_STATE, ptr addrspace(1) %rnn.future.ptr, align RECIPE_STATE_ALIGN
%rnn.gate.model = load double, ptr addrspace(1) %rnn.gate.ptr, align 8 %rnn.gate = call RECIPE_STATE @recipe.decode(double %rnn.gate.model) %rnn.dh = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %rnn.dy, RECIPE_STATE %rnn.future)
%rnn.square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %rnn.gate, RECIPE_STATE %rnn.gate) %rnn.tanh.derivative = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %rnn.square)
%rnn.coded.relu = icmp eq i32 %cell.activation, 1 %rnn.coded.tanh = icmp eq i32 %cell.activation, 2 %rnn.coded.sigmoid = icmp eq i32 %cell.activation, 3
%rnn.positive = call i1 @recipe.ogt(double %rnn.gate.model, double 0.0) %rnn.relu.derivative = select i1 %rnn.positive, RECIPE_STATE %state.one, RECIPE_STATE %state.zero
%rnn.one.minus = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %rnn.gate) %rnn.sigmoid.derivative = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %rnn.gate, RECIPE_STATE %rnn.one.minus)
%rnn.coded.a = select i1 %rnn.coded.relu, RECIPE_STATE %rnn.relu.derivative, RECIPE_STATE %state.one
%rnn.coded.b = select i1 %rnn.coded.tanh, RECIPE_STATE %rnn.tanh.derivative, RECIPE_STATE %rnn.coded.a
%rnn.coded.derivative = select i1 %rnn.coded.sigmoid, RECIPE_STATE %rnn.sigmoid.derivative, RECIPE_STATE %rnn.coded.b
%rnn.derivative = select i1 %coded, RECIPE_STATE %rnn.coded.derivative, RECIPE_STATE %rnn.tanh.derivative
%rnn.delta = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %rnn.dh, RECIPE_STATE %rnn.derivative) %rnn.delta.index = add i32 %delta.base, %rnn.index %scan.rnn.delta.index.wide = add i64 %scan.delta.base.wide, %scan.rnn.index.wide
%rnn.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.rnn.delta.index.wide
store RECIPE_STATE %rnn.delta, ptr addrspace(1) %rnn.delta.ptr, align RECIPE_STATE_ALIGN %rnn.next = add i32 %rnn.hidden, 1
br label %rnn.delta.loop gru.delta.loop: %gru.hidden = phi i32 [ 0, %rnn.check ], [ %gru.next, %gru.delta.step ]
%gru.more = icmp ult i32 %gru.hidden, %out.channels
br i1 %gru.more, label %gru.delta.step, label %gru.reset.loop gru.delta.step:
%gru.hidden.wide = zext i32 %gru.hidden to i64 %gru.hidden.base = mul i32 %gru.hidden, %length %scan.gru.hidden.base.wide = mul i64 %gru.hidden.wide, %scan.length.wide %gru.local = add i32 %gru.hidden.base, %time.current %scan.gru.local.wide = add i64 %scan.gru.hidden.base.wide, %scan.time.current.wide
%gru.index = add i32 %row.output.base, %gru.local %scan.gru.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.local.wide %gru.previous.local = add i32 %gru.hidden.base, %previous.safe %scan.gru.previous.local.wide = add i64 %scan.gru.hidden.base.wide, %scan.previous.safe.wide
%gru.previous.index = add i32 %row.output.base, %gru.previous.local %scan.gru.previous.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.previous.local.wide
%gru.dy.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %scan.gru.index.wide %gru.future.index = add i32 %dh.start, %gru.hidden %scan.gru.future.index.wide = add i64 %scan.dh.start.wide, %gru.hidden.wide
%gru.future.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.gru.future.index.wide
%gru.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %scan.gru.previous.index.wide
%gru.z.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.gru.index.wide
%gru.n.index = add i32 %gru.index, %gate2.batch %scan.gru.n.index.wide = add i64 %scan.gru.index.wide, %scan.gate2.batch.wide
%gru.n.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.gru.n.index.wide
%gru.dy = load RECIPE_STATE, ptr addrspace(1) %gru.dy.ptr, align RECIPE_STATE_ALIGN
%gru.future = load RECIPE_STATE, ptr addrspace(1) %gru.future.ptr, align RECIPE_STATE_ALIGN
%gru.previous.loaded.model = load double, ptr addrspace(1) %gru.previous.ptr, align 8 %gru.previous.loaded = call RECIPE_STATE @recipe.decode(double %gru.previous.loaded.model)
%gru.previous = select i1 %previous.exists, RECIPE_STATE %gru.previous.loaded, RECIPE_STATE %state.zero
%gru.z.model = load double, ptr addrspace(1) %gru.z.ptr, align 8 %gru.z = call RECIPE_STATE @recipe.decode(double %gru.z.model)
%gru.n.model = load double, ptr addrspace(1) %gru.n.ptr, align 8 %gru.n = call RECIPE_STATE @recipe.decode(double %gru.n.model) %gru.dh = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %gru.dy, RECIPE_STATE %gru.future)
%gru.one.z = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %gru.z) %gru.z.difference = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %gru.previous, RECIPE_STATE %gru.n)
%gru.dz.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.dh, RECIPE_STATE %gru.z.difference) %gru.dz.1 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.dz.0, RECIPE_STATE %gru.z)
%gru.dz = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.dz.1, RECIPE_STATE %gru.one.z) %gru.n.square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.n, RECIPE_STATE %gru.n)
%gru.n.derivative = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %gru.n.square) %gru.dn.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.dh, RECIPE_STATE %gru.one.z)
%gru.dn = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.dn.0, RECIPE_STATE %gru.n.derivative) %gru.dz.index = add i32 %delta.base, %gru.index %scan.gru.dz.index.wide = add i64 %scan.delta.base.wide, %scan.gru.index.wide
%gru.dn.index.0 = add i32 %delta.base, %gate2.batch %scan.gru.dn.base.wide = add i64 %scan.delta.base.wide, %scan.gate2.batch.wide %scan.gru.dn.index.wide = add i64 %scan.gru.dn.base.wide, %scan.gru.index.wide
%gru.dz.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.gru.dz.index.wide
%gru.dn.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.gru.dn.index.wide
store RECIPE_STATE %gru.dz, ptr addrspace(1) %gru.dz.ptr, align RECIPE_STATE_ALIGN
store RECIPE_STATE %gru.dn, ptr addrspace(1) %gru.dn.ptr, align RECIPE_STATE_ALIGN %gru.next = add i32 %gru.hidden, 1
br label %gru.delta.loop gru.reset.loop:
%gru.source = phi i32 [ 0, %gru.delta.loop ], [ %gru.source.next, %gru.reset.store ]
%gru.source.more = icmp ult i32 %gru.source, %out.channels %scan.gru.source.wide = zext i32 %gru.source to i64
br i1 %gru.source.more, label %gru.reset.sum.loop, label %delta.done gru.reset.sum.loop:
%gru.target = phi i32 [ 0, %gru.reset.loop ], [ %gru.target.next, %gru.reset.sum.step ]
%gru.reset.sum = phi RECIPE_STATE [ %state.zero, %gru.reset.loop ], [ %gru.reset.sum.next, %gru.reset.sum.step ]
%gru.target.more = icmp ult i32 %gru.target, %out.channels
br i1 %gru.target.more, label %gru.reset.sum.step, label %gru.reset.store gru.reset.sum.step:
%scan.gru.target.wide = zext i32 %gru.target to i64 %gru.candidate.base = mul i32 %gate.stride, 2 %scan.gru.candidate.base.wide = mul i64 %scan.gate.stride.wide, 2 %gru.candidate.state = add i32 %gru.candidate.base, %gate.stride.0 %scan.gru.candidate.state.wide = add i64 %scan.gru.candidate.base.wide, %scan.gate.stride.0.wide
%gru.weight.row = mul i32 %gru.source, %out.channels %scan.gru.weight.row.wide = mul i64 %scan.gru.source.wide, %scan.out.channels.wide %gru.weight.local = add i32 %gru.weight.row, %gru.target %scan.gru.weight.local.wide = add i64 %scan.gru.weight.row.wide, %scan.gru.target.wide
%gru.weight.index = add i32 %gru.candidate.state, %gru.weight.local %scan.gru.weight.index.wide = add i64 %scan.gru.candidate.state.wide, %scan.gru.weight.local.wide
%gru.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %scan.gru.weight.index.wide
%gru.target.base = mul i32 %gru.target, %length %scan.gru.target.base.wide = mul i64 %scan.gru.target.wide, %scan.length.wide %gru.target.local = add i32 %gru.target.base, %time.current %scan.gru.target.local.wide = add i64 %scan.gru.target.base.wide, %scan.time.current.wide
%gru.target.index = add i32 %row.output.base, %gru.target.local %scan.gru.target.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.target.local.wide %gru.target.delta.0 = add i32 %delta.base, %gate2.batch %scan.gru.target.delta.0.wide = add i64 %scan.delta.base.wide, %scan.gate2.batch.wide
%gru.target.delta.index = add i32 %gru.target.delta.0, %gru.target.index %scan.gru.target.delta.index.wide = add i64 %scan.gru.target.delta.0.wide, %scan.gru.target.index.wide
%gru.target.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.gru.target.delta.index.wide
%gru.weight.model = load double, ptr addrspace(1) %gru.weight.ptr, align 8 %gru.weight = call RECIPE_STATE @recipe.decode(double %gru.weight.model)
%gru.target.delta = load RECIPE_STATE, ptr addrspace(1) %gru.target.delta.ptr, align RECIPE_STATE_ALIGN
%gru.reset.product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.weight, RECIPE_STATE %gru.target.delta)
%gru.reset.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %gru.reset.sum, RECIPE_STATE %gru.reset.product)
%gru.target.next = add i32 %gru.target, 1 br label %gru.reset.sum.loop gru.reset.store:
%gru.source.base = mul i32 %gru.source, %length %scan.gru.source.base.wide = mul i64 %scan.gru.source.wide, %scan.length.wide %gru.source.local = add i32 %gru.source.base, %time.current %scan.gru.source.local.wide = add i64 %scan.gru.source.base.wide, %scan.time.current.wide
%gru.source.index = add i32 %row.output.base, %gru.source.local %scan.gru.source.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.source.local.wide
%gru.source.previous.local = add i32 %gru.source.base, %previous.safe %scan.gru.source.previous.local.wide = add i64 %scan.gru.source.base.wide, %scan.previous.safe.wide
%gru.source.previous.index = add i32 %row.output.base, %gru.source.previous.local %scan.gru.source.previous.index.wide = add i64 %scan.row.output.base.wide, %scan.gru.source.previous.local.wide
%gru.source.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %scan.gru.source.previous.index.wide
%gru.r.index = add i32 %batch, %gru.source.index %scan.gru.r.index.wide = add i64 %scan.batch.wide, %scan.gru.source.index.wide
%gru.r.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.gru.r.index.wide
%gru.source.previous.loaded.model = load double, ptr addrspace(1) %gru.source.previous.ptr, align 8 %gru.source.previous.loaded = call RECIPE_STATE @recipe.decode(double %gru.source.previous.loaded.model)
%gru.source.previous = select i1 %previous.exists, RECIPE_STATE %gru.source.previous.loaded, RECIPE_STATE %state.zero
%gru.r.model = load double, ptr addrspace(1) %gru.r.ptr, align 8 %gru.r = call RECIPE_STATE @recipe.decode(double %gru.r.model)
%gru.dr = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.reset.sum, RECIPE_STATE %gru.source.previous) %gru.one.r = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %gru.r)
%gru.dr.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.dr, RECIPE_STATE %gru.r) %gru.dr.1 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %gru.dr.0, RECIPE_STATE %gru.one.r)
%gru.dr.base = add i32 %delta.base, %batch %scan.gru.dr.base.wide = add i64 %scan.delta.base.wide, %scan.batch.wide %gru.dr.index = add i32 %gru.dr.base, %gru.source.index %scan.gru.dr.index.wide = add i64 %scan.gru.dr.base.wide, %scan.gru.source.index.wide
%gru.dr.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.gru.dr.index.wide
store RECIPE_STATE %gru.dr.1, ptr addrspace(1) %gru.dr.ptr, align RECIPE_STATE_ALIGN %gru.source.next = add i32 %gru.source, 1
br label %gru.reset.loop gate.delta.loop: %hidden = phi i32 [ 0, %scan.mode ], [ %hidden.next, %gate.delta.step ]
%hidden.more = icmp ult i32 %hidden, %out.channels br i1 %hidden.more, label %gate.delta.step, label %delta.done
gate.delta.step: %hidden.wide = zext i32 %hidden to i64 %hidden.base = mul i32 %hidden, %length %scan.lstm.hidden.base.wide = mul i64 %hidden.wide, %scan.length.wide %local = add i32 %hidden.base, %time.current %scan.lstm.local.wide = add i64 %scan.lstm.hidden.base.wide, %scan.time.current.wide
%index = add i32 %row.output.base, %local %scan.lstm.index.wide = add i64 %scan.row.output.base.wide, %scan.lstm.local.wide %previous.local = add i32 %hidden.base, %previous.safe %scan.lstm.previous.local.wide = add i64 %scan.lstm.hidden.base.wide, %scan.previous.safe.wide
%previous.index = add i32 %row.output.base, %previous.local %scan.lstm.previous.index.wide = add i64 %scan.row.output.base.wide, %scan.lstm.previous.local.wide %cell.base = mul i32 %gates, %batch %scan.lstm.cell.base.wide = mul i64 %scan.gates.wide, %scan.batch.wide
%cell.index = add i32 %cell.base, %index %scan.lstm.cell.index.wide = add i64 %scan.lstm.cell.base.wide, %scan.lstm.index.wide %cell.previous.index = add i32 %cell.base, %previous.index %scan.lstm.cell.previous.index.wide = add i64 %scan.lstm.cell.base.wide, %scan.lstm.previous.index.wide
%dy.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %scan.lstm.index.wide %dh.index = add i32 %dh.start, %hidden %scan.lstm.dh.index.wide = add i64 %scan.dh.start.wide, %hidden.wide
%dc.index = add i32 %dc.start, %hidden %scan.lstm.dc.index.wide = add i64 %scan.dc.start.wide, %hidden.wide %dh.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.lstm.dh.index.wide
%dc.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.lstm.dc.index.wide
%cell.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.lstm.cell.index.wide
%cell.previous.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.lstm.cell.previous.index.wide
%dy = load RECIPE_STATE, ptr addrspace(1) %dy.ptr, align RECIPE_STATE_ALIGN %dh.future = load RECIPE_STATE, ptr addrspace(1) %dh.ptr, align RECIPE_STATE_ALIGN
%dc.future = load RECIPE_STATE, ptr addrspace(1) %dc.ptr, align RECIPE_STATE_ALIGN %cell.model = load double, ptr addrspace(1) %cell.ptr, align 8 %cell = call RECIPE_STATE @recipe.decode(double %cell.model)
%cell.previous.loaded.model = load double, ptr addrspace(1) %cell.previous.ptr, align 8 %cell.previous.loaded = call RECIPE_STATE @recipe.decode(double %cell.previous.loaded.model)
%cell.previous = select i1 %previous.exists, RECIPE_STATE %cell.previous.loaded, RECIPE_STATE %state.zero
%i.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.lstm.index.wide %f.index = add i32 %batch, %index %scan.lstm.f.index.wide = add i64 %scan.batch.wide, %scan.lstm.index.wide
%o.index = add i32 %f.index, %batch %scan.lstm.o.index.wide = add i64 %scan.lstm.f.index.wide, %scan.batch.wide %g.index = add i32 %o.index, %batch %scan.lstm.g.index.wide = add i64 %scan.lstm.o.index.wide, %scan.batch.wide
%f.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.lstm.f.index.wide
%o.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.lstm.o.index.wide
%g.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.lstm.g.index.wide
%i.model = load double, ptr addrspace(1) %i.ptr, align 8 %i = call RECIPE_STATE @recipe.decode(double %i.model) %f.model = load double, ptr addrspace(1) %f.ptr, align 8 %f = call RECIPE_STATE @recipe.decode(double %f.model)
%o.model = load double, ptr addrspace(1) %o.ptr, align 8 %o = call RECIPE_STATE @recipe.decode(double %o.model) %g.model = load double, ptr addrspace(1) %g.ptr, align 8 %g = call RECIPE_STATE @recipe.decode(double %g.model)
%dh = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %dy, RECIPE_STATE %dh.future) %cell.tanh = call RECIPE_STATE @recipe.state.tanh(RECIPE_STATE %cell)
%cell.tanh.square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %cell.tanh, RECIPE_STATE %cell.tanh) %cell.tanh.derivative = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %cell.tanh.square)
%cell.chain.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dh, RECIPE_STATE %o) %cell.chain = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %cell.chain.0, RECIPE_STATE %cell.tanh.derivative)
%dc = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %dc.future, RECIPE_STATE %cell.chain) %one.o = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %o) %do.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dh, RECIPE_STATE %cell.tanh)
%do.1 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %do.0, RECIPE_STATE %o) %do = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %do.1, RECIPE_STATE %one.o) %one.i = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %i) %di.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dc, RECIPE_STATE %g)
%di.1 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %di.0, RECIPE_STATE %i) %di = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %di.1, RECIPE_STATE %one.i) %one.f = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %f)
%df.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dc, RECIPE_STATE %cell.previous) %df.1 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %df.0, RECIPE_STATE %f) %df = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %df.1, RECIPE_STATE %one.f)
%g.square = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %g, RECIPE_STATE %g) %one.g.square = call RECIPE_STATE @recipe.state.sub(RECIPE_STATE %state.one, RECIPE_STATE %g.square) %dg.0 = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dc, RECIPE_STATE %i)
%dg = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dg.0, RECIPE_STATE %one.g.square) %dc.previous = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %dc, RECIPE_STATE %f)
store RECIPE_STATE %dc.previous, ptr addrspace(1) %dc.ptr, align RECIPE_STATE_ALIGN %delta0.index = add i32 %delta.base, %index %scan.lstm.delta0.index.wide = add i64 %scan.delta.base.wide, %scan.lstm.index.wide
%delta1.index = add i32 %delta0.index, %batch %scan.lstm.delta1.index.wide = add i64 %scan.lstm.delta0.index.wide, %scan.batch.wide %delta2.index = add i32 %delta1.index, %batch %scan.lstm.delta2.index.wide = add i64 %scan.lstm.delta1.index.wide, %scan.batch.wide
%delta3.index = add i32 %delta2.index, %batch %scan.lstm.delta3.index.wide = add i64 %scan.lstm.delta2.index.wide, %scan.batch.wide
%delta0.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.lstm.delta0.index.wide
%delta1.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.lstm.delta1.index.wide
%delta2.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.lstm.delta2.index.wide
%delta3.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.lstm.delta3.index.wide
store RECIPE_STATE %di, ptr addrspace(1) %delta0.ptr, align RECIPE_STATE_ALIGN store RECIPE_STATE %df, ptr addrspace(1) %delta1.ptr, align RECIPE_STATE_ALIGN
store RECIPE_STATE %do, ptr addrspace(1) %delta2.ptr, align RECIPE_STATE_ALIGN store RECIPE_STATE %dg, ptr addrspace(1) %delta3.ptr, align RECIPE_STATE_ALIGN
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
%gate.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.delta.index.wide
%gate.delta = load RECIPE_STATE, ptr addrspace(1) %gate.delta.ptr, align RECIPE_STATE_ALIGN
%state.hidden.base = mul i32 %source.channel, %length %scan.state.hidden.base.wide = mul i64 %scan.source.channel.wide, %scan.length.wide %state.local = add i32 %state.hidden.base, %previous.safe %scan.state.local.wide = add i64 %scan.state.hidden.base.wide, %scan.previous.safe.wide %state.index.value = add i32 %row.output.base, %state.local %scan.state.index.value.wide = add i64 %scan.row.output.base.wide, %scan.state.local.wide
%state.ptr = getelementptr inbounds double, ptr addrspace(1) %output, i64 %scan.state.index.value.wide
%state.loaded.model = load double, ptr addrspace(1) %state.ptr, align 8 %state.loaded = call RECIPE_STATE @recipe.decode(double %state.loaded.model)
%state.value = select i1 %previous.exists, RECIPE_STATE %state.loaded, RECIPE_STATE %state.zero
%candidate.gate = icmp eq i32 %gate, 2 %gru.candidate = and i1 %gru, %candidate.gate
%parameter.reset.local = add i32 %state.hidden.base, %time.current %scan.parameter.reset.local.wide = add i64 %scan.state.hidden.base.wide, %scan.time.current.wide
%parameter.reset.row = add i32 %row.output.base, %parameter.reset.local %scan.parameter.reset.row.wide = add i64 %scan.row.output.base.wide, %scan.parameter.reset.local.wide
%parameter.reset.raw = add i32 %batch, %parameter.reset.row %scan.parameter.reset.raw.wide = add i64 %scan.batch.wide, %scan.parameter.reset.row.wide
%parameter.reset.index = select i1 %gru.candidate, i32 %parameter.reset.raw, i32 0 %scan.parameter.reset.index.wide = select i1 %gru.candidate, i64 %scan.parameter.reset.raw.wide, i64 0
%parameter.reset.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.parameter.reset.index.wide
%parameter.reset.model = load double, ptr addrspace(1) %parameter.reset.ptr, align 8 %parameter.reset = call RECIPE_STATE @recipe.decode(double %parameter.reset.model)
%parameter.reset.state = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %parameter.reset, RECIPE_STATE %state.value)
%parameter.state = select i1 %gru.candidate, RECIPE_STATE %parameter.reset.state, RECIPE_STATE %state.value
%source.value = select i1 %state.weight, RECIPE_STATE %parameter.state, RECIPE_STATE %state.one
%contribution = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %source.value, RECIPE_STATE %gate.delta) %scan.p.wide = zext i32 %p to i64 %scan.row.gradient.index.wide = add i64 %scan.row.gradient.start.wide, %scan.p.wide %row.gradient.index = add i32 %row.gradient.start, %p
%row.gradient.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.row.gradient.index.wide
%row.gradient.old = load RECIPE_STATE, ptr addrspace(1) %row.gradient.ptr, align RECIPE_STATE_ALIGN
%row.gradient.new = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %row.gradient.old, RECIPE_STATE %contribution)
store RECIPE_STATE %row.gradient.new, ptr addrspace(1) %row.gradient.ptr, align RECIPE_STATE_ALIGN
br label %parameter.advance parameter.advance:
%p.next = add nuw i32 %p, 1 br label %parameter.loop hidden.gradient.loop:
%state.channel = phi i32 [ 0, %parameter.loop ], [ %state.channel.next, %hidden.gradient.store ] %scan.state.channel.wide = zext i32 %state.channel to i64
%state.channel.more = icmp ult i32 %state.channel, %out.channels
br i1 %state.channel.more, label %hidden.gradient.sum.loop, label %time.done hidden.gradient.sum.loop:
%state.term = phi i32 [ 0, %hidden.gradient.loop ], [ %state.term.next, %hidden.gradient.sum.step ]
%state.sum = phi RECIPE_STATE [ %state.zero, %hidden.gradient.loop ], [ %state.sum.next, %hidden.gradient.sum.step ]
%state.terms = mul i32 %gates, %out.channels %state.term.more = icmp ult i32 %state.term, %state.terms
br i1 %state.term.more, label %hidden.gradient.sum.step, label %hidden.gradient.store hidden.gradient.sum.step:
%state.gate = udiv i32 %state.term, %out.channels %state.hidden = urem i32 %state.term, %out.channels %scan.state.gate.wide = zext i32 %state.gate to i64 %scan.state.hidden.wide = zext i32 %state.hidden to i64
%state.gate.base = mul i32 %state.gate, %gate.stride %scan.state.gate.base.wide = mul i64 %scan.state.gate.wide, %scan.gate.stride.wide %state.matrix.base = add i32 %state.gate.base, %gate.stride.0 %scan.state.matrix.base.wide = add i64 %scan.state.gate.base.wide, %scan.gate.stride.0.wide
%state.weight.row = mul i32 %state.channel, %out.channels %scan.state.weight.row.wide = mul i64 %scan.state.channel.wide, %scan.out.channels.wide %state.weight.local = add i32 %state.weight.row, %state.hidden %scan.state.weight.local.wide = add i64 %scan.state.weight.row.wide, %scan.state.hidden.wide
%state.weight.index = add i32 %state.matrix.base, %state.weight.local %scan.state.weight.index.wide = add i64 %scan.state.matrix.base.wide, %scan.state.weight.local.wide
%state.weight.ptr = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %scan.state.weight.index.wide
%state.delta.hidden.base = mul i32 %state.hidden, %length %scan.state.delta.hidden.base.wide = mul i64 %scan.state.hidden.wide, %scan.length.wide
%state.delta.local = add i32 %state.delta.hidden.base, %time.current %scan.state.delta.local.wide = add i64 %scan.state.delta.hidden.base.wide, %scan.time.current.wide
%state.delta.row = add i32 %row.output.base, %state.delta.local %scan.state.delta.row.wide = add i64 %scan.row.output.base.wide, %scan.state.delta.local.wide %state.delta.gate.base = mul i32 %state.gate, %batch %scan.state.delta.gate.base.wide = mul i64 %scan.state.gate.wide, %scan.batch.wide
%state.delta.base = add i32 %delta.base, %state.delta.gate.base %scan.state.delta.base.wide = add i64 %scan.delta.base.wide, %scan.state.delta.gate.base.wide
%state.delta.index = add i32 %state.delta.base, %state.delta.row %scan.state.delta.index.wide = add i64 %scan.state.delta.base.wide, %scan.state.delta.row.wide
%state.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.state.delta.index.wide
%state.weight.value.model = load double, ptr addrspace(1) %state.weight.ptr, align 8 %state.weight.value = call RECIPE_STATE @recipe.decode(double %state.weight.value.model)
%state.delta.value = load RECIPE_STATE, ptr addrspace(1) %state.delta.ptr, align RECIPE_STATE_ALIGN
%state.product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %state.weight.value, RECIPE_STATE %state.delta.value) %state.candidate = icmp eq i32 %state.gate, 2
%state.gru.candidate = and i1 %gru, %state.candidate %state.reset.hidden.base = mul i32 %state.channel, %length %scan.state.reset.hidden.base.wide = mul i64 %scan.state.channel.wide, %scan.length.wide
%state.reset.local = add i32 %state.reset.hidden.base, %time.current %scan.state.reset.local.wide = add i64 %scan.state.reset.hidden.base.wide, %scan.time.current.wide
%state.reset.row = add i32 %row.output.base, %state.reset.local %scan.state.reset.row.wide = add i64 %scan.row.output.base.wide, %scan.state.reset.local.wide %state.reset.raw = add i32 %batch, %state.reset.row %scan.state.reset.raw.wide = add i64 %scan.batch.wide, %scan.state.reset.row.wide
%state.reset.index = select i1 %state.gru.candidate, i32 %state.reset.raw, i32 0 %scan.state.reset.index.wide = select i1 %state.gru.candidate, i64 %scan.state.reset.raw.wide, i64 0
%state.reset.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.state.reset.index.wide
%state.reset.model = load double, ptr addrspace(1) %state.reset.ptr, align 8 %state.reset = call RECIPE_STATE @recipe.decode(double %state.reset.model)
%state.reset.product = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %state.product, RECIPE_STATE %state.reset)
%state.contribution = select i1 %state.gru.candidate, RECIPE_STATE %state.reset.product, RECIPE_STATE %state.product
%state.sum.next = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %state.sum, RECIPE_STATE %state.contribution) %state.term.next = add nuw i32 %state.term, 1
br label %hidden.gradient.sum.loop hidden.gradient.store:
%state.dh.index = add i32 %dh.start, %state.channel %scan.state.dh.index.wide = add i64 %scan.dh.start.wide, %scan.state.channel.wide
%state.dh.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %scan.state.dh.index.wide
%state.direct.hidden.base = mul i32 %state.channel, %length %scan.state.direct.hidden.base.wide = mul i64 %scan.state.channel.wide, %scan.length.wide
%state.direct.local = add i32 %state.direct.hidden.base, %time.current %scan.state.direct.local.wide = add i64 %scan.state.direct.hidden.base.wide, %scan.time.current.wide
%state.direct.index = add i32 %row.output.base, %state.direct.local %scan.state.direct.index.wide = add i64 %scan.row.output.base.wide, %scan.state.direct.local.wide
%state.direct.delta.ptr = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %delta, i64 %scan.state.direct.index.wide
%state.direct.z.ptr = getelementptr inbounds double, ptr addrspace(1) %context, i64 %scan.state.direct.index.wide
%state.direct.dy = load RECIPE_STATE, ptr addrspace(1) %state.direct.delta.ptr, align RECIPE_STATE_ALIGN
%state.direct.future = load RECIPE_STATE, ptr addrspace(1) %state.dh.ptr, align RECIPE_STATE_ALIGN
%state.direct.z.model = load double, ptr addrspace(1) %state.direct.z.ptr, align 8 %state.direct.z = call RECIPE_STATE @recipe.decode(double %state.direct.z.model)
%state.direct.dh = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %state.direct.dy, RECIPE_STATE %state.direct.future)
%state.direct.raw = call RECIPE_STATE @recipe.state.mul(RECIPE_STATE %state.direct.z, RECIPE_STATE %state.direct.dh)
%state.direct = select i1 %gru, RECIPE_STATE %state.direct.raw, RECIPE_STATE %state.zero
%state.total = call RECIPE_STATE @recipe.state.add(RECIPE_STATE %state.sum, RECIPE_STATE %state.direct)
store RECIPE_STATE %state.total, ptr addrspace(1) %state.dh.ptr, align RECIPE_STATE_ALIGN %state.channel.next = add nuw i32 %state.channel, 1
br label %hidden.gradient.loop time.done: br label %time.loop row.done: %row.next = add i32 %row, %threads
br label %row.loop reduce.entry: call void @grid_barrier(i32 %threads)
call void @reduce_rows_state(ptr addrspace(1) %backward, ptr addrspace(1) %gradient, i32 %rows, i32 %parameters, i32 %parameters, i32 %row.gradient.base, i32 %offset, i32 %threads)
br label %projection.entry
projection.entry: call void @grid_barrier(i32 %threads) br label %projection.loop projection.loop:
%projection.gate = phi i32 [ 0, %projection.entry ], [ %projection.next, %projection.step ]
%projection.more = icmp ult i32 %projection.gate, %gates
br i1 %projection.more, label %projection.step, label %exit projection.step:
%projection.gate.wide = zext i32 %projection.gate to i64 %projection.weight.offset = mul i32 %projection.gate, %gate.stride %projection.weight.offset.wide = mul i64 %projection.gate.wide, %scan.gate.stride.wide
%projection.weights = getelementptr inbounds double, ptr addrspace(1) %weights, i64 %projection.weight.offset.wide
%projection.delta.gate = mul i32 %projection.gate, %batch %projection.delta.gate.wide = mul i64 %projection.gate.wide, %scan.batch.wide
%projection.delta.offset = add i32 %delta.base, %projection.delta.gate %projection.delta.offset.wide = add i64 %scan.delta.base.wide, %projection.delta.gate.wide
%projection.delta = getelementptr inbounds RECIPE_STATE, ptr addrspace(1) %backward, i64 %projection.delta.offset.wide
%projection.gradient.offset = add i32 %offset, %projection.weight.offset
call void @contraction_reverse_body( ptr addrspace(1) %input, ptr addrspace(1) %projection.weights, ptr addrspace(1) %output,
ptr addrspace(1) %projection.delta, ptr addrspace(1) %previous, ptr addrspace(1) %gradient, i1 %write.input, i1 false, i1 false, i1 false,
i32 %rows, i32 %in.channels, i32 %length, i32 %out.channels, i32 %length, i32 0,
i32 %projection.gradient.offset, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k,
i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) call void @grid_barrier(i32 %threads) %projection.next = add i32 %projection.gate, 1 br label %projection.loop
invalid: call void @llvm.trap() br label %exit exit: ret void } attributes #0 = { nounwind "amdgpu-flat-work-group-size"="RECIPE_WORKGROUP_SIZE,RECIPE_WORKGROUP_SIZE" } attributes #1 = { alwaysinline nounwind } attributes #3 = { noinline nounwind }
; Fully unroll the product loop so each insertelement uses a constant lane index
; and the accumulator vector remains in registers.
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
