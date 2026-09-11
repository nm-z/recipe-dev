**rules:**
data must start and end up back on d1
one FLOP:
	a * b
	a + b
4-byte numbers, choose opt in 4-byte step
link speed between 2 d's = avg(d(speed),d(speed))
concurrently, any d can:
	calc(di, XB) -> result
	transfer(di, XB) -> df

**definitions:**
Field	Meaning
p	Result bytes produced globally
e	Result bytes present on d1, the required final destination
rN	Total bytes currently resident on device N
cNb	Cumulative result bytes calculated by device N
tNb	Cumulative bytes transferred to device N
cNt	Seconds remaining on active device N’s calculation
t1t	Seconds remaining on active device N's transfer

**system:**
op: a * b = y
d(speed,rate)
d1(d1s, d1r)
d2(d2s, d2r)

**state:**
in:
	system:
		d1s,	d1r,	d2s,	d2r
		40,	2,	80,	10
	state:
		r1, r2, p, e, c1b, c1t, c2b, c2t, t1b, t1t, t2b, t2t
		40, 0,  0, 0, 0,   0,   0,   0,   0,   0,   0,   0
out:
	ti,	tf,	tb,		cb,		,c
	d1|d2,d1|d2,[4,40],	[4,40],	none|d1|d2|both
