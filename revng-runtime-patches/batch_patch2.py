"""Transform LLVM 16's PromoteMemoryToRegister.cpp so that mem2reg computes liveness and phi placement for all
general-path allocas together on compact CFG and dominator rows, and summarizes heavily used allocas in one sweep of
the function instead of walking each one's uses. Phi insertion order, naming and renaming are unchanged."""
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace("""  void ComputeLiveInBlocks(AllocaInst *AI, AllocaInfo &Info,
                           const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
                           SmallPtrSetImpl<BasicBlock *> &LiveInBlocks);""",
"""  void ComputeLiveInBlocks(AllocaInst *AI, AllocaInfo &Info,
                           const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
                           SmallPtrSetImpl<BasicBlock *> &LiveInBlocks);

  /// Allocas with at least this many uses are summarized by one sweep of the
  /// function instead of walking their use lists several times each.
  static constexpr unsigned SummaryThreshold = 1024;

  /// General-path allocas whose liveness and phi placement are computed
  /// together, one bit each in a word per block.
  static constexpr unsigned GroupSize = 64;
  static_assert(GroupSize <= 64, "one 64-bit word per block");

  /// What one sweep of the function learns about a heavily used alloca.
  struct AllocaSummary {
    AllocaInfo Info;
    /// Numbers of the blocks that store to the alloca and of those whose
    /// first access to it is a load.
    SmallVector<unsigned, 0> DefiningBlockNumbers;
    SmallVector<unsigned, 0> ExposedBlockNumbers;
    /// Whether every use is a load from or a store into the alloca.
    bool OnlyLoadsAndStores = true;
    unsigned LastBlock = ~0U;
    bool StoredInLastBlock = false;
    bool ExposedInLastBlock = false;
  };
  DenseMap<AllocaInst *, unsigned> SummaryIndex;
  std::vector<AllocaSummary> Summaries;

  /// The CFG and dominator tree over block numbers: predecessors and
  /// successors in compressed rows, as predecessors() and successors() list
  /// them including duplicate edges, and each block's dominator tree level,
  /// DFS entry number and children; unreachable blocks have level ~0U.
  std::vector<BasicBlock *> BlockOf;
  std::vector<unsigned> PredecessorStart, Predecessors;
  std::vector<unsigned> SuccessorStart, Successors;
  std::vector<unsigned> DomLevel, DomDFSIn, DomChildStart, DomChildren;

  /// A general-path alloca whose phi nodes are placed after all allocas were
  /// classified, in the same order.
  struct DeferredAlloca {
    unsigned AllocaNum;
    /// Blocks that store to the alloca and blocks whose first access to it
    /// is a load, when not taken from a summary.
    SmallVector<unsigned, 4> DefiningBlockNumbers;
    SmallVector<unsigned, 4> ExposedBlockNumbers;
    int Summary;
  };
  std::vector<DeferredAlloca> Deferred;

  void summarizeHeavyAllocas(Function &F);
  void buildCompactRows(Function &F,
                        std::vector<std::pair<unsigned, unsigned>> Edges);
  void deferAlloca(unsigned AllocaNum, AllocaInst *AI, const AllocaInfo &Info,
                   int Summary);
  void computePhiBlocks(ArrayRef<const DeferredAlloca *> Group,
                        std::vector<SmallVector<unsigned, 32>> &PhiBlocks);
  void placeDeferredPhis(Function &F);""")

replace("""  AllocaInfo Info;
  LargeBlockInfo LBI;
  ForwardIDFCalculator IDF(DT);

  for (unsigned AllocaNum = 0; AllocaNum != Allocas.size(); ++AllocaNum) {
    AllocaInst *AI = Allocas[AllocaNum];

    assert(isAllocaPromotable(AI) && "Cannot promote non-promotable alloca!");
    assert(AI->getParent()->getParent() == &F &&
           "All allocas should be in the same function, which is same as DF!");

    removeIntrinsicUsers(AI);
""",
"""  AllocaInfo Info;
  LargeBlockInfo LBI;

  summarizeHeavyAllocas(F);

  for (unsigned AllocaNum = 0; AllocaNum != Allocas.size(); ++AllocaNum) {
    AllocaInst *AI = Allocas[AllocaNum];

    // A summarized alloca has only loads and stores as uses, so it has no
    // intrinsic users to remove; it was checked when it was selected.
    auto Found = SummaryIndex.find(AI);
    int Summary = Found != SummaryIndex.end()
                      && Summaries[Found->second].OnlyLoadsAndStores
                    ? int(Found->second)
                    : -1;

    if (Summary < 0) {
      assert(isAllocaPromotable(AI) && "Cannot promote non-promotable alloca!");
      removeIntrinsicUsers(AI);
    }
    assert(AI->getParent()->getParent() == &F &&
           "All allocas should be in the same function, which is same as DF!");
""")

replace("""    // Calculate the set of read and write-locations for each alloca.  This is
    // analogous to finding the 'uses' and 'definitions' of each variable.
    Info.AnalyzeAlloca(AI);""",
"""    // Calculate the set of read and write-locations for each alloca.  This is
    // analogous to finding the 'uses' and 'definitions' of each variable.
    if (Summary >= 0)
      Info = Summaries[Summary].Info;
    else
      Info.AnalyzeAlloca(AI);""")

replace("""    // Keep the reverse mapping of the 'Allocas' array for the rename pass.
    AllocaLookup[Allocas[AllocaNum]] = AllocaNum;

    // Unique the set of defining blocks for efficient lookup.
    SmallPtrSet<BasicBlock *, 32> DefBlocks(Info.DefiningBlocks.begin(),
                                            Info.DefiningBlocks.end());

    // Determine which blocks the value is live in.  These are blocks which lead
    // to uses.
    SmallPtrSet<BasicBlock *, 32> LiveInBlocks;
    ComputeLiveInBlocks(AI, Info, DefBlocks, LiveInBlocks);

    // At this point, we're committed to promoting the alloca using IDF's, and
    // the standard SSA construction algorithm.  Determine which blocks need phi
    // nodes and see if we can optimize out some work by avoiding insertion of
    // dead phi nodes.
    IDF.setLiveInBlocks(LiveInBlocks);
    IDF.setDefiningBlocks(DefBlocks);
    SmallVector<BasicBlock *, 32> PHIBlocks;
    IDF.calculate(PHIBlocks);
    llvm::sort(PHIBlocks, [this](BasicBlock *A, BasicBlock *B) {
      return BBNumbers.find(A)->second < BBNumbers.find(B)->second;
    });

    unsigned CurrentVersion = 0;
    for (BasicBlock *BB : PHIBlocks)
      QueuePhiNode(BB, AllocaNum, CurrentVersion);
  }

  if (Allocas.empty())
    return; // All of the allocas must have been trivial!
""",
"""    // Keep the reverse mapping of the 'Allocas' array for the rename pass.
    AllocaLookup[Allocas[AllocaNum]] = AllocaNum;

    // Liveness and phi placement depend only on this alloca's own loads and
    // stores and on the CFG, which the remaining classification leaves
    // unchanged; compute them afterwards for all such allocas together, and
    // queue the phi nodes in this order.
    deferAlloca(AllocaNum, AI, Info, Summary);
  }

  placeDeferredPhis(F);

  if (Allocas.empty())
    return; // All of the allocas must have been trivial!
""")

replace("""/// Determine which blocks the value is live in.""",
"""void PromoteMem2Reg::summarizeHeavyAllocas(Function &F) {
  if (BBNumbers.empty()) {
    unsigned ID = 0;
    for (auto &BB : F)
      BBNumbers[&BB] = ID++;
  }

  for (AllocaInst *AI : Allocas) {
    if (AI->isUsedByMetadata() || AI->hasMetadata(LLVMContext::MD_DIAssignID)
        || !AI->hasNUsesOrMore(SummaryThreshold))
      continue;
    SummaryIndex[AI] = Summaries.size();
    Summaries.emplace_back();
    Summaries.back().Info.clear();
  }
  if (Summaries.empty())
    return;

  // One sweep collects the summaries and the CFG edges.
  std::vector<std::pair<unsigned, unsigned>> Edges;
  for (BasicBlock &BB : F) {
    unsigned Number = BBNumbers.find(&BB)->second;
    for (Instruction &I : BB) {
      for (Use &Operand : I.operands()) {
        auto *AI = dyn_cast<AllocaInst>(Operand.get());
        if (AI == nullptr)
          continue;
        auto Found = SummaryIndex.find(AI);
        if (Found == SummaryIndex.end())
          continue;
        AllocaSummary &S = Summaries[Found->second];
        if (S.LastBlock != Number) {
          S.LastBlock = Number;
          S.StoredInLastBlock = false;
          S.ExposedInLastBlock = false;
        }
        if (auto *LI = dyn_cast<LoadInst>(&I);
            LI != nullptr
            && Operand.getOperandNo() == LoadInst::getPointerOperandIndex()) {
          S.Info.UsingBlocks.push_back(&BB);
          if (!S.StoredInLastBlock && !S.ExposedInLastBlock) {
            S.ExposedInLastBlock = true;
            S.ExposedBlockNumbers.push_back(Number);
          }
        } else if (auto *SI = dyn_cast<StoreInst>(&I);
                   SI != nullptr
                   && Operand.getOperandNo()
                        == StoreInst::getPointerOperandIndex()) {
          S.Info.DefiningBlocks.push_back(&BB);
          S.Info.OnlyStore = SI;
          if (!S.StoredInLastBlock)
            S.DefiningBlockNumbers.push_back(Number);
          S.StoredInLastBlock = true;
        } else {
          S.OnlyLoadsAndStores = false;
          continue;
        }
        if (S.Info.OnlyUsedInOneBlock) {
          if (!S.Info.OnlyBlock)
            S.Info.OnlyBlock = &BB;
          else if (S.Info.OnlyBlock != &BB)
            S.Info.OnlyUsedInOneBlock = false;
        }
      }
    }
    for (BasicBlock *Successor : successors(&BB))
      Edges.emplace_back(Number, BBNumbers.find(Successor)->second);
  }
  buildCompactRows(F, std::move(Edges));
}

/// Build the compact rows from (block, successor) edges recorded block by
/// block in successor order, or from the function when none were recorded.
void PromoteMem2Reg::buildCompactRows(
    Function &F, std::vector<std::pair<unsigned, unsigned>> Edges) {
  unsigned NumBlocks = BBNumbers.size();
  BlockOf.resize(NumBlocks);
  for (auto &BB : F)
    BlockOf[BBNumbers.find(&BB)->second] = &BB;
  if (Edges.empty())
    for (unsigned B = 0; B < NumBlocks; ++B)
      for (BasicBlock *Successor : successors(BlockOf[B]))
        Edges.emplace_back(B, BBNumbers.find(Successor)->second);

  SuccessorStart.assign(NumBlocks + 1, 0);
  PredecessorStart.assign(NumBlocks + 1, 0);
  for (auto &[From, To] : Edges) {
    ++SuccessorStart[From + 1];
    ++PredecessorStart[To + 1];
  }
  for (unsigned I = 0; I < NumBlocks; ++I) {
    SuccessorStart[I + 1] += SuccessorStart[I];
    PredecessorStart[I + 1] += PredecessorStart[I];
  }
  Successors.resize(Edges.size());
  Predecessors.resize(Edges.size());
  std::vector<unsigned> Next(PredecessorStart.begin(), PredecessorStart.end() - 1);
  for (unsigned I = 0; I < Edges.size(); ++I) {
    Successors[I] = Edges[I].second;
    Predecessors[Next[Edges[I].second]++] = Edges[I].first;
  }

  DT.updateDFSNumbers();
  DomLevel.assign(NumBlocks, ~0U);
  DomDFSIn.assign(NumBlocks, 0);
  DomChildStart.assign(NumBlocks + 1, 0);
  for (unsigned B = 0; B < NumBlocks; ++B) {
    DomChildStart[B + 1] = DomChildStart[B];
    DomTreeNode *Node = DT.getNode(BlockOf[B]);
    if (Node == nullptr)
      continue;
    DomLevel[B] = Node->getLevel();
    DomDFSIn[B] = Node->getDFSNumIn();
    for (DomTreeNode *Child : Node->children()) {
      DomChildren.push_back(BBNumbers.find(Child->getBlock())->second);
      ++DomChildStart[B + 1];
    }
  }
}

/// Record the defining blocks and the blocks where the alloca is read before
/// it is written, the seeds of ComputeLiveInBlocks.
void PromoteMem2Reg::deferAlloca(unsigned AllocaNum, AllocaInst *AI,
                                 const AllocaInfo &Info, int Summary) {
  DeferredAlloca &D = Deferred.emplace_back();
  D.AllocaNum = AllocaNum;
  D.Summary = Summary;
  if (Summary >= 0)
    return;

  SmallPtrSet<BasicBlock *, 32> DefBlocks(Info.DefiningBlocks.begin(),
                                          Info.DefiningBlocks.end());
  for (BasicBlock *BB : DefBlocks)
    D.DefiningBlockNumbers.push_back(BBNumbers.find(BB)->second);
  SmallPtrSet<BasicBlock *, 32> Seen;
  for (BasicBlock *BB : Info.UsingBlocks) {
    if (!Seen.insert(BB).second)
      continue;
    bool Exposed = true;
    if (DefBlocks.count(BB)) {
      // As in ComputeLiveInBlocks: the first access in the block decides.
      for (Instruction &I : *BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I); SI && SI->getOperand(1) == AI) {
          Exposed = false;
          break;
        }
        if (auto *LI = dyn_cast<LoadInst>(&I); LI && LI->getOperand(0) == AI)
          break;
      }
    }
    if (Exposed)
      D.ExposedBlockNumbers.push_back(BBNumbers.find(BB)->second);
  }
}

/// For up to GroupSize deferred allocas, compute where each needs a phi node:
/// the blocks where it is live in, as ComputeLiveInBlocks finds them, then
/// IDFCalculatorBase::calculate's iterated dominance frontier restricted to
/// them, both evaluated on the compact rows instead of the IR.
void PromoteMem2Reg::computePhiBlocks(
    ArrayRef<const DeferredAlloca *> Group,
    std::vector<SmallVector<unsigned, 32>> &PhiBlocks) {
  assert(Group.size() <= GroupSize);
  unsigned NumBlocks = BlockOf.size();
  auto Defining = [&](const DeferredAlloca *D) -> ArrayRef<unsigned> {
    if (D->Summary >= 0)
      return Summaries[D->Summary].DefiningBlockNumbers;
    return D->DefiningBlockNumbers;
  };
  auto Exposed = [&](const DeferredAlloca *D) -> ArrayRef<unsigned> {
    if (D->Summary >= 0)
      return Summaries[D->Summary].ExposedBlockNumbers;
    return D->ExposedBlockNumbers;
  };

  std::vector<uint64_t> Defines(NumBlocks), Live(NumBlocks), Pending(NumBlocks);
  std::vector<unsigned> Worklist;
  for (unsigned K = 0; K < Group.size(); ++K) {
    uint64_t Bit = uint64_t(1) << K;
    for (unsigned B : Defining(Group[K]))
      Defines[B] |= Bit;
    for (unsigned B : Exposed(Group[K])) {
      if (Pending[B] == 0)
        Worklist.push_back(B);
      Pending[B] |= Bit;
    }
  }
  while (!Worklist.empty()) {
    unsigned B = Worklist.back();
    Worklist.pop_back();
    uint64_t Bits = Pending[B] & ~Live[B];
    Pending[B] = 0;
    if (Bits == 0)
      continue;
    Live[B] |= Bits;
    for (unsigned I = PredecessorStart[B]; I < PredecessorStart[B + 1]; ++I) {
      unsigned P = Predecessors[I];
      uint64_t Add = Bits & ~Defines[P] & ~Live[P];
      if (Add == 0)
        continue;
      if (Pending[P] == 0)
        Worklist.push_back(P);
      Pending[P] |= Add;
    }
  }

  using Entry = std::pair<unsigned, std::pair<unsigned, unsigned>>;
  std::vector<unsigned> VisitedPQ(NumBlocks, ~0U), VisitedWorklist(NumBlocks, ~0U);
  PhiBlocks.assign(Group.size(), {});
  for (unsigned K = 0; K < Group.size(); ++K) {
    uint64_t Bit = uint64_t(1) << K;
    std::priority_queue<Entry, SmallVector<Entry, 32>, less_second> PQ;
    for (unsigned B : Defining(Group[K]))
      if (DomLevel[B] != ~0U) {
        PQ.push({B, {DomLevel[B], DomDFSIn[B]}});
        VisitedWorklist[B] = K;
      }
    while (!PQ.empty()) {
      unsigned Root = PQ.top().first;
      unsigned RootLevel = PQ.top().second.first;
      PQ.pop();
      Worklist.assign(1, Root);
      while (!Worklist.empty()) {
        unsigned Node = Worklist.back();
        Worklist.pop_back();
        for (unsigned I = SuccessorStart[Node]; I < SuccessorStart[Node + 1];
             ++I) {
          unsigned Succ = Successors[I];
          if (DomLevel[Succ] > RootLevel || VisitedPQ[Succ] == K)
            continue;
          VisitedPQ[Succ] = K;
          if (!(Live[Succ] & Bit))
            continue;
          PhiBlocks[K].push_back(Succ);
          if (!(Defines[Succ] & Bit))
            PQ.push({Succ, {DomLevel[Succ], DomDFSIn[Succ]}});
        }
        for (unsigned I = DomChildStart[Node]; I < DomChildStart[Node + 1];
             ++I) {
          unsigned Child = DomChildren[I];
          if (VisitedWorklist[Child] != K) {
            VisitedWorklist[Child] = K;
            Worklist.push_back(Child);
          }
        }
      }
    }
    // Block numbers follow BBNumbers, the order phi nodes are queued in.
    llvm::sort(PhiBlocks[K]);
  }
}

void PromoteMem2Reg::placeDeferredPhis(Function &F) {
  if (Deferred.empty())
    return;
  if (BlockOf.empty())
    buildCompactRows(F, {});

  for (unsigned Start = 0; Start < Deferred.size(); Start += GroupSize) {
    SmallVector<const DeferredAlloca *, GroupSize> Group;
    for (unsigned I = Start; I < Deferred.size() && I < Start + GroupSize; ++I)
      Group.push_back(&Deferred[I]);
    std::vector<SmallVector<unsigned, 32>> PhiBlocks;
    computePhiBlocks(Group, PhiBlocks);
    for (unsigned K = 0; K < Group.size(); ++K) {
      unsigned CurrentVersion = 0;
      for (unsigned B : PhiBlocks[K])
        QueuePhiNode(BlockOf[B], Group[K]->AllocaNum, CurrentVersion);
    }
  }
  Deferred.clear();
}

/// Determine which blocks the value is live in.""")

replace('#include "llvm/Transforms/Utils/PromoteMemToReg.h"\n', '#include "llvm/Transforms/Utils/PromoteMemToReg.h"\n#include <queue>\n')
Path(sys.argv[2]).write_text(source)
print('patched')
