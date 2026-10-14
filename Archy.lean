set_option autoImplicit false
-- numbers
inductive QiFp where          -- exp man sgn
  | uint (bits : Nat)         -- int2
  | int (bits : Nat)          -- int8
  | float (exp man : Nat)     -- bf or fp
  | qi (bits block : Nat)     -- qi
  | iq (bits block : Nat)     -- iq
  deriving DecidableEq, Repr

def QiFp.bits : QiFp → Nat    -- quantizations and precisions
  | .uint n | .int n => n     -- int
  | .float e m => 1 + e + m   -- fp
  | .qi n _ | .iq n _ => n    -- qi
-- instructions
inductive Calc where          -- calculation instructions
  | FFMA | FMUL | FADD | FMNMX | MUFU | FSETP
  | XMAD | IADD | IADD3 | ISCADD | IMNMX | ICMP | SEL
  | LOP3 | SHL | SHR | SHF | PRMT | BFE | BFI | POPC | FLO
  deriving DecidableEq, Repr

inductive Conv where          -- conversion instructions
  | I2F | F2I | F2F | I2I
  deriving DecidableEq, Repr

inductive Op where            -- 2 instruction types
  | calc (c : Calc)
  | conv (c : Conv)
  deriving DecidableEq, Repr

structure Instruction where   -- actions
  op : Op
-- locations
structure Location where
  locsize : Nat               -- bits
  locname : String            -- place
  deriving DecidableEq
-- links
structure Link where
  lnksrc  : Location          -- origin
  lnkdst  : Location          -- target
-- units
structure Data where
  dtype   : QiFp              -- encoding
  dname   : String            -- association
  dnum    : BitVec dtype.bits -- bitstring
  dstate  : Location          -- place
