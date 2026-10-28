helper='''/// Whether \\p From branches to \\p To. The successors of \\p From and the
/// predecessors of \\p To are walked alternately, so the cost is bounded by
/// twice the shorter list: a block with millions of successors or
/// predecessors, such as a dispatcher, does not have to be walked in full.
static bool hasEdge(const BasicBlock *From, const BasicBlock *To) {
  auto Successor = succ_begin(From), SuccessorEnd = succ_end(From);
  auto Predecessor = pred_begin(To), PredecessorEnd = pred_end(To);
  while (true) {
    if (Successor == SuccessorEnd)
      return false;
    if (*Successor == To)
      return true;
    ++Successor;

    if (Predecessor == PredecessorEnd)
      return false;
    if (*Predecessor == From)
      return true;
    ++Predecessor;
  }
}

'''
p='patched/b/llvm/lib/Analysis/DomTreeUpdater.cpp'; s=open(p).read()
old='''bool DomTreeUpdater::isUpdateValid('''
assert s.count(old)==1
s=s.replace(old, helper+old,1)
old='''  const bool HasEdge = llvm::is_contained(successors(From), To);'''
assert s.count(old)==1
s=s.replace(old,'''  const bool HasEdge = hasEdge(From, To);''')
open(p,'w').write(s)
p='patched/b/llvm/lib/Transforms/Utils/Local.cpp'; s=open(p).read()
old='''bool llvm::TryToSimplifyUncondBranchFromEmptyBlock('''
assert s.count(old)==1
s=s.replace(old, helper+old,1)
old='''    // To avoid processing the same predecessor more than once.
    SmallPtrSet<BasicBlock *, 8> SeenPreds;
    // All predecessors of BB will be moved to Succ.
    SmallPtrSet<BasicBlock *, 8> PredsOfSucc(pred_begin(Succ), pred_end(Succ));
    Updates.reserve(Updates.size() + 2 * pred_size(BB) + 1);
    for (auto *PredOfBB : predecessors(BB))
      // This predecessor of BB may already have Succ as a successor.
      if (!PredsOfSucc.contains(PredOfBB))
        if (SeenPreds.insert(PredOfBB).second)
          Updates.push_back({DominatorTree::Insert, PredOfBB, Succ});'''
assert s.count(old)==1
s=s.replace(old,'''    // To avoid processing the same predecessor more than once.
    SmallPtrSet<BasicBlock *, 8> SeenPreds;
    // All predecessors of BB will be moved to Succ.
    Updates.reserve(Updates.size() + 2 * pred_size(BB) + 1);
    for (auto *PredOfBB : predecessors(BB))
      // This predecessor of BB may already have Succ as a successor.
      if (SeenPreds.insert(PredOfBB).second && !hasEdge(PredOfBB, Succ))
        Updates.push_back({DominatorTree::Insert, PredOfBB, Succ});''')
open(p,'w').write(s)
