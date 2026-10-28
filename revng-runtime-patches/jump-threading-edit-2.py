p='patched/b/llvm/lib/Analysis/DomTreeUpdater.cpp'; s=open(p).read()
s=s.replace('#include "llvm/IR/Constants.h"\n','#include "llvm/IR/CFG.h"\n#include "llvm/IR/Constants.h"\n',1)
open(p,'w').write(s)
