//===- PromoteMemoryToRegister.cpp - Convert allocas to registers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file promotes memory references to be register references.  It promotes
// alloca instructions which only have loads and stores as uses.  An alloca is
// transformed by using iterated dominator frontiers to place PHI nodes, then
// traversing the function in depth-first order to rewrite loads and stores as
// appropriate.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugInfoPreservation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <queue>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mem2reg"

STATISTIC(NumLocalPromoted, "Number of alloca's promoted within one block");
STATISTIC(NumSingleStore,   "Number of alloca's promoted with a single store");
STATISTIC(NumDeadAlloca,    "Number of dead alloca's removed");
STATISTIC(NumPHIInsert,     "Number of PHI nodes inserted");

bool llvm::isAllocaPromotable(const AllocaInst *AI) {
  // Only allow direct and non-volatile loads and stores...
  for (const User *U : AI->users()) {
    if (const LoadInst *LI = dyn_cast<LoadInst>(U)) {
      // Note that atomic loads can be transformed; atomic semantics do
      // not have any meaning for a local alloca.
      if (LI->isVolatile() || LI->getType() != AI->getAllocatedType())
        return false;
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getValueOperand() == AI ||
          SI->getValueOperand()->getType() != AI->getAllocatedType())
        return false; // Don't allow a store OF the AI, only INTO the AI.
      // Note that atomic stores can be transformed; atomic semantics do
      // not have any meaning for a local alloca.
      if (SI->isVolatile())
        return false;
    } else if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
      if (!II->isLifetimeStartOrEnd() && !II->isDroppable())
        return false;
    } else if (const BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      if (!onlyUsedByLifetimeMarkersOrDroppableInsts(BCI))
        return false;
    } else if (const GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(U)) {
      if (!GEPI->hasAllZeroIndices())
        return false;
      if (!onlyUsedByLifetimeMarkersOrDroppableInsts(GEPI))
        return false;
    } else if (const AddrSpaceCastInst *ASCI = dyn_cast<AddrSpaceCastInst>(U)) {
      if (!onlyUsedByLifetimeMarkers(ASCI))
        return false;
    } else {
      return false;
    }
  }

  return true;
}

namespace {

/// Helper for updating assignment tracking debug info when promoting allocas.
class AssignmentTrackingInfo {
  /// DbgAssignIntrinsics linked to the alloca with at most one per variable
  /// fragment. (i.e. not be a comprehensive set if there are multiple
  /// dbg.assigns for one variable fragment).
  SmallVector<DbgVariableIntrinsic *> DbgAssigns;

public:
  void init(AllocaInst *AI) {
    SmallSet<DebugVariable, 2> Vars;
    for (DbgAssignIntrinsic *DAI : at::getAssignmentMarkers(AI)) {
      if (Vars.insert(DebugVariable(DAI)).second)
        DbgAssigns.push_back(DAI);
    }
  }

  /// Update assignment tracking debug info given for the to-be-deleted store
  /// \p ToDelete that stores to this alloca.
  void updateForDeletedStore(StoreInst *ToDelete, DIBuilder &DIB) const {
    // There's nothing to do if the alloca doesn't have any variables using
    // assignment tracking.
    if (DbgAssigns.empty()) {
      assert(at::getAssignmentMarkers(ToDelete).empty());
      return;
    }

    // Just leave dbg.assign intrinsics in place and remember that we've seen
    // one for each variable fragment.
    SmallSet<DebugVariable, 2> VarHasDbgAssignForStore;
    for (DbgAssignIntrinsic *DAI : at::getAssignmentMarkers(ToDelete))
      VarHasDbgAssignForStore.insert(DebugVariable(DAI));

    // It's possible for variables using assignment tracking to have no
    // dbg.assign linked to this store. These are variables in DbgAssigns that
    // are missing from VarHasDbgAssignForStore. Since there isn't a dbg.assign
    // to mark the assignment - and the store is going to be deleted - insert a
    // dbg.value to do that now. An untracked store may be either one that
    // cannot be represented using assignment tracking (non-const offset or
    // size) or one that is trackable but has had its DIAssignID attachment
    // dropped accidentally.
    for (auto *DAI : DbgAssigns) {
      if (VarHasDbgAssignForStore.contains(DebugVariable(DAI)))
        continue;
      ConvertDebugDeclareToDebugValue(DAI, ToDelete, DIB);
    }
  }

  /// Update assignment tracking debug info given for the newly inserted PHI \p
  /// NewPhi.
  void updateForNewPhi(PHINode *NewPhi, DIBuilder &DIB) const {
    // Regardless of the position of dbg.assigns relative to stores, the
    // incoming values into a new PHI should be the same for the (imaginary)
    // debug-phi.
    for (auto *DAI : DbgAssigns)
      ConvertDebugDeclareToDebugValue(DAI, NewPhi, DIB);
  }

  void clear() { DbgAssigns.clear(); }
  bool empty() { return DbgAssigns.empty(); }
};

struct AllocaInfo {
  using DbgUserVec = SmallVector<DbgVariableIntrinsic *, 1>;

  SmallVector<BasicBlock *, 32> DefiningBlocks;
  SmallVector<BasicBlock *, 32> UsingBlocks;

  StoreInst *OnlyStore;
  BasicBlock *OnlyBlock;
  bool OnlyUsedInOneBlock;

  /// Debug users of the alloca - does not include dbg.assign intrinsics.
  DbgUserVec DbgUsers;
  /// Helper to update assignment tracking debug info.
  AssignmentTrackingInfo AssignmentTracking;

  void clear() {
    DefiningBlocks.clear();
    UsingBlocks.clear();
    OnlyStore = nullptr;
    OnlyBlock = nullptr;
    OnlyUsedInOneBlock = true;
    DbgUsers.clear();
    AssignmentTracking.clear();
  }

  /// Scan the uses of the specified alloca, filling in the AllocaInfo used
  /// by the rest of the pass to reason about the uses of this alloca.
  void AnalyzeAlloca(AllocaInst *AI) {
    clear();

    // As we scan the uses of the alloca instruction, keep track of stores,
    // and decide whether all of the loads and stores to the alloca are within
    // the same basic block.
    for (User *U : AI->users()) {
      Instruction *User = cast<Instruction>(U);

      if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
        // Remember the basic blocks which define new values for the alloca
        DefiningBlocks.push_back(SI->getParent());
        OnlyStore = SI;
      } else {
        LoadInst *LI = cast<LoadInst>(User);
        // Otherwise it must be a load instruction, keep track of variable
        // reads.
        UsingBlocks.push_back(LI->getParent());
      }

      if (OnlyUsedInOneBlock) {
        if (!OnlyBlock)
          OnlyBlock = User->getParent();
        else if (OnlyBlock != User->getParent())
          OnlyUsedInOneBlock = false;
      }
    }
    DbgUserVec AllDbgUsers;
    findDbgUsers(AllDbgUsers, AI);
    std::copy_if(AllDbgUsers.begin(), AllDbgUsers.end(),
                 std::back_inserter(DbgUsers), [](DbgVariableIntrinsic *DII) {
                   return !isa<DbgAssignIntrinsic>(DII);
                 });
    AssignmentTracking.init(AI);
  }
};

/// Data package used by RenamePass().
struct RenamePassData {
  using ValVector = std::vector<Value *>;
  using LocationVector = std::vector<DebugLoc>;

  RenamePassData(BasicBlock *B, BasicBlock *P, ValVector V, LocationVector L)
      : BB(B), Pred(P), Values(std::move(V)), Locations(std::move(L)) {}

  BasicBlock *BB;
  BasicBlock *Pred;
  ValVector Values;
  LocationVector Locations;
};

/// This assigns and keeps a per-bb relative ordering of load/store
/// instructions in the block that directly load or store an alloca.
///
/// This functionality is important because it avoids scanning large basic
/// blocks multiple times when promoting many allocas in the same block.
class LargeBlockInfo {
  /// For each instruction that we track, keep the index of the
  /// instruction.
  ///
  /// The index starts out as the number of the instruction from the start of
  /// the block.
  DenseMap<const Instruction *, unsigned> InstNumbers;

public:

  /// This code only looks at accesses to allocas.
  static bool isInterestingInstruction(const Instruction *I) {
    return (isa<LoadInst>(I) && isa<AllocaInst>(I->getOperand(0))) ||
           (isa<StoreInst>(I) && isa<AllocaInst>(I->getOperand(1)));
  }

  /// Get or calculate the index of the specified instruction.
  unsigned getInstructionIndex(const Instruction *I) {
    assert(isInterestingInstruction(I) &&
           "Not a load/store to/from an alloca?");

    // If we already have this instruction number, return it.
    DenseMap<const Instruction *, unsigned>::iterator It = InstNumbers.find(I);
    if (It != InstNumbers.end())
      return It->second;

    // Scan the whole block to get the instruction.  This accumulates
    // information for every interesting instruction in the block, in order to
    // avoid gratuitus rescans.
    const BasicBlock *BB = I->getParent();
    unsigned InstNo = 0;
    for (const Instruction &BBI : *BB)
      if (isInterestingInstruction(&BBI))
        InstNumbers[&BBI] = InstNo++;
    It = InstNumbers.find(I);

    assert(It != InstNumbers.end() && "Didn't insert instruction?");
    return It->second;
  }

  void deleteValue(const Instruction *I) { InstNumbers.erase(I); }

  void clear() { InstNumbers.clear(); }
};

struct PromoteMem2Reg {
  /// The alloca instructions being promoted.
  std::vector<AllocaInst *> Allocas;

  DominatorTree &DT;
  DIBuilder DIB;

  /// A cache of @llvm.assume intrinsics used by SimplifyInstruction.
  AssumptionCache *AC;

  const SimplifyQuery SQ;

  /// Reverse mapping of Allocas.
  DenseMap<AllocaInst *, unsigned> AllocaLookup;

  /// The PhiNodes we're adding.
  ///
  /// That map is used to simplify some Phi nodes as we iterate over it, so
  /// it should have deterministic iterators.  We could use a MapVector, but
  /// since we already maintain a map from BasicBlock* to a stable numbering
  /// (BBNumbers), the DenseMap is more efficient (also supports removal).
  DenseMap<std::pair<unsigned, unsigned>, PHINode *> NewPhiNodes;

  /// For each PHI node, keep track of which entry in Allocas it corresponds
  /// to.
  DenseMap<PHINode *, unsigned> PhiToAllocaMap;

  /// For each alloca, we keep track of the dbg.declare intrinsic that
  /// describes it, if any, so that we can convert it to a dbg.value
  /// intrinsic if the alloca gets promoted.
  SmallVector<AllocaInfo::DbgUserVec, 8> AllocaDbgUsers;

  /// For each alloca, keep an instance of a helper class that gives us an easy
  /// way to update assignment tracking debug info if the alloca is promoted.
  SmallVector<AssignmentTrackingInfo, 8> AllocaATInfo;

  /// The set of basic blocks the renamer has already visited.
  SmallPtrSet<BasicBlock *, 16> Visited;

  /// Contains a stable numbering of basic blocks to avoid non-determinstic
  /// behavior.
  DenseMap<BasicBlock *, unsigned> BBNumbers;

  /// Lazily compute the number of predecessors a block has.
  DenseMap<const BasicBlock *, unsigned> BBNumPreds;

public:
  PromoteMem2Reg(ArrayRef<AllocaInst *> Allocas, DominatorTree &DT,
                 AssumptionCache *AC)
      : Allocas(Allocas.begin(), Allocas.end()), DT(DT),
        DIB(*DT.getRoot()->getParent()->getParent(), /*AllowUnresolved*/ false),
        AC(AC), SQ(DT.getRoot()->getParent()->getParent()->getDataLayout(),
                   nullptr, &DT, AC) {}

  void run();

private:
  void RemoveFromAllocasList(unsigned &AllocaIdx) {
    Allocas[AllocaIdx] = Allocas.back();
    Allocas.pop_back();
    --AllocaIdx;
  }

  unsigned getNumPreds(const BasicBlock *BB) {
    unsigned &NP = BBNumPreds[BB];
    if (NP == 0)
      NP = pred_size(BB) + 1;
    return NP - 1;
  }

  void ComputeLiveInBlocks(AllocaInst *AI, AllocaInfo &Info,
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
  void placeDeferredPhis(Function &F);
  void RenamePass(BasicBlock *BB, BasicBlock *Pred,
                  RenamePassData::ValVector &IncVals,
                  RenamePassData::LocationVector &IncLocs,
                  std::vector<RenamePassData> &Worklist);
  bool QueuePhiNode(BasicBlock *BB, unsigned AllocaIdx, unsigned &Version);
};

} // end anonymous namespace

/// Given a LoadInst LI this adds assume(LI != null) after it.
static void addAssumeNonNull(AssumptionCache *AC, LoadInst *LI) {
  Function *AssumeIntrinsic =
      Intrinsic::getDeclaration(LI->getModule(), Intrinsic::assume);
  ICmpInst *LoadNotNull = new ICmpInst(ICmpInst::ICMP_NE, LI,
                                       Constant::getNullValue(LI->getType()));
  LoadNotNull->insertAfter(LI);
  CallInst *CI = CallInst::Create(AssumeIntrinsic, {LoadNotNull});
  CI->insertAfter(LoadNotNull);
  AC->registerAssumption(cast<AssumeInst>(CI));
}

static void convertMetadataToAssumes(LoadInst *LI, Value *Val,
                                     const DataLayout &DL, AssumptionCache *AC,
                                     const DominatorTree *DT) {
  // If the load was marked as nonnull we don't want to lose that information
  // when we erase this Load. So we preserve it with an assume. As !nonnull
  // returns poison while assume violations are immediate undefined behavior,
  // we can only do this if the value is known non-poison.
  if (AC && LI->getMetadata(LLVMContext::MD_nonnull) &&
      LI->getMetadata(LLVMContext::MD_noundef) &&
      !isKnownNonZero(Val, DL, 0, AC, LI, DT))
    addAssumeNonNull(AC, LI);
}

static void removeIntrinsicUsers(AllocaInst *AI) {
  // Knowing that this alloca is promotable, we know that it's safe to kill all
  // instructions except for load and store.

  for (Use &U : llvm::make_early_inc_range(AI->uses())) {
    Instruction *I = cast<Instruction>(U.getUser());
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      continue;

    // Drop the use of AI in droppable instructions.
    if (I->isDroppable()) {
      I->dropDroppableUse(U);
      continue;
    }

    if (!I->getType()->isVoidTy()) {
      // The only users of this bitcast/GEP instruction are lifetime intrinsics.
      // Follow the use/def chain to erase them now instead of leaving it for
      // dead code elimination later.
      for (Use &UU : llvm::make_early_inc_range(I->uses())) {
        Instruction *Inst = cast<Instruction>(UU.getUser());

        // Drop the use of I in droppable instructions.
        if (Inst->isDroppable()) {
          Inst->dropDroppableUse(UU);
          continue;
        }
        Inst->eraseFromParent();
      }
    }
    I->eraseFromParent();
  }
}

/// Rewrite as many loads as possible given a single store.
///
/// When there is only a single store, we can use the domtree to trivially
/// replace all of the dominated loads with the stored value. Do so, and return
/// true if this has successfully promoted the alloca entirely. If this returns
/// false there were some loads which were not dominated by the single store
/// and thus must be phi-ed with undef. We fall back to the standard alloca
/// promotion algorithm in that case.
static bool rewriteSingleStoreAlloca(AllocaInst *AI, AllocaInfo &Info,
                                     LargeBlockInfo &LBI, const DataLayout &DL,
                                     DominatorTree &DT, AssumptionCache *AC) {
  StoreInst *OnlyStore = Info.OnlyStore;
  bool StoringGlobalVal = !isa<Instruction>(OnlyStore->getOperand(0));
  BasicBlock *StoreBB = OnlyStore->getParent();
  int StoreIndex = -1;

  // Clear out UsingBlocks.  We will reconstruct it here if needed.
  Info.UsingBlocks.clear();

  for (User *U : make_early_inc_range(AI->users())) {
    Instruction *UserInst = cast<Instruction>(U);
    if (UserInst == OnlyStore)
      continue;
    LoadInst *LI = cast<LoadInst>(UserInst);

    // Okay, if we have a load from the alloca, we want to replace it with the
    // only value stored to the alloca.  We can do this if the value is
    // dominated by the store.  If not, we use the rest of the mem2reg machinery
    // to insert the phi nodes as needed.
    if (!StoringGlobalVal) { // Non-instructions are always dominated.
      if (LI->getParent() == StoreBB) {
        // If we have a use that is in the same block as the store, compare the
        // indices of the two instructions to see which one came first.  If the
        // load came before the store, we can't handle it.
        if (StoreIndex == -1)
          StoreIndex = LBI.getInstructionIndex(OnlyStore);

        if (unsigned(StoreIndex) > LBI.getInstructionIndex(LI)) {
          // Can't handle this load, bail out.
          Info.UsingBlocks.push_back(StoreBB);
          continue;
        }
      } else if (!DT.dominates(StoreBB, LI->getParent())) {
        // If the load and store are in different blocks, use BB dominance to
        // check their relationships.  If the store doesn't dom the use, bail
        // out.
        Info.UsingBlocks.push_back(LI->getParent());
        continue;
      }
    }

    // Otherwise, we *can* safely rewrite this load.
    Value *ReplVal = OnlyStore->getOperand(0);
    // If the replacement value is the load, this must occur in unreachable
    // code.
    if (ReplVal == LI)
      ReplVal = PoisonValue::get(LI->getType());

    convertMetadataToAssumes(LI, ReplVal, DL, AC, &DT);
    LI->replaceAllUsesWith(ReplVal);
    LI->eraseFromParent();
    LBI.deleteValue(LI);
  }

  // Finally, after the scan, check to see if the store is all that is left.
  if (!Info.UsingBlocks.empty())
    return false; // If not, we'll have to fall back for the remainder.

  DIBuilder DIB(*AI->getModule(), /*AllowUnresolved*/ false);
  // Update assignment tracking info for the store we're going to delete.
  Info.AssignmentTracking.updateForDeletedStore(Info.OnlyStore, DIB);

  // Record debuginfo for the store and remove the declaration's
  // debuginfo.
  for (DbgVariableIntrinsic *DII : Info.DbgUsers) {
    if (DII->isAddressOfVariable()) {
      ConvertDebugDeclareToDebugValue(DII, Info.OnlyStore, DIB);
      DII->eraseFromParent();
    } else if (DII->getExpression()->startsWithDeref()) {
      DII->eraseFromParent();
    }
  }

  // Remove dbg.assigns linked to the alloca as these are now redundant.
  at::deleteAssignmentMarkers(AI);

  // Remove the (now dead) store and alloca.
  Info.OnlyStore->eraseFromParent();
  LBI.deleteValue(Info.OnlyStore);

  AI->eraseFromParent();
  return true;
}

/// Many allocas are only used within a single basic block.  If this is the
/// case, avoid traversing the CFG and inserting a lot of potentially useless
/// PHI nodes by just performing a single linear pass over the basic block
/// using the Alloca.
///
/// If we cannot promote this alloca (because it is read before it is written),
/// return false.  This is necessary in cases where, due to control flow, the
/// alloca is undefined only on some control flow paths.  e.g. code like
/// this is correct in LLVM IR:
///  // A is an alloca with no stores so far
///  for (...) {
///    int t = *A;
///    if (!first_iteration)
///      use(t);
///    *A = 42;
///  }
static bool promoteSingleBlockAlloca(AllocaInst *AI, const AllocaInfo &Info,
                                     LargeBlockInfo &LBI,
                                     const DataLayout &DL,
                                     DominatorTree &DT,
                                     AssumptionCache *AC) {
  // The trickiest case to handle is when we have large blocks. Because of this,
  // this code is optimized assuming that large blocks happen.  This does not
  // significantly pessimize the small block case.  This uses LargeBlockInfo to
  // make it efficient to get the index of various operations in the block.

  // Walk the use-def list of the alloca, getting the locations of all stores.
  using StoresByIndexTy = SmallVector<std::pair<unsigned, StoreInst *>, 64>;
  StoresByIndexTy StoresByIndex;

  for (User *U : AI->users())
    if (StoreInst *SI = dyn_cast<StoreInst>(U))
      StoresByIndex.push_back(std::make_pair(LBI.getInstructionIndex(SI), SI));

  // Sort the stores by their index, making it efficient to do a lookup with a
  // binary search.
  llvm::sort(StoresByIndex, less_first());

  // Walk all of the loads from this alloca, replacing them with the nearest
  // store above them, if any.
  for (User *U : make_early_inc_range(AI->users())) {
    LoadInst *LI = dyn_cast<LoadInst>(U);
    if (!LI)
      continue;

    unsigned LoadIdx = LBI.getInstructionIndex(LI);

    // Find the nearest store that has a lower index than this load.
    StoresByIndexTy::iterator I = llvm::lower_bound(
        StoresByIndex,
        std::make_pair(LoadIdx, static_cast<StoreInst *>(nullptr)),
        less_first());
    Value *ReplVal;
    if (I == StoresByIndex.begin()) {
      if (StoresByIndex.empty())
        // If there are no stores, the load takes the undef value.
        ReplVal = UndefValue::get(LI->getType());
      else
        // There is no store before this load, bail out (load may be affected
        // by the following stores - see main comment).
        return false;
    } else {
      // Otherwise, there was a store before this load, the load takes its
      // value.
      ReplVal = std::prev(I)->second->getOperand(0);
    }

    convertMetadataToAssumes(LI, ReplVal, DL, AC, &DT);

    // If the replacement value is the load, this must occur in unreachable
    // code.
    if (ReplVal == LI)
      ReplVal = PoisonValue::get(LI->getType());

    LI->replaceAllUsesWith(ReplVal);
    LI->eraseFromParent();
    LBI.deleteValue(LI);
  }

  // Remove the (now dead) stores and alloca.
  DIBuilder DIB(*AI->getModule(), /*AllowUnresolved*/ false);
  while (!AI->use_empty()) {
    StoreInst *SI = cast<StoreInst>(AI->user_back());
    // Update assignment tracking info for the store we're going to delete.
    Info.AssignmentTracking.updateForDeletedStore(SI, DIB);
    // Record debuginfo for the store before removing it.
    for (DbgVariableIntrinsic *DII : Info.DbgUsers) {
      if (DII->isAddressOfVariable()) {
        ConvertDebugDeclareToDebugValue(DII, SI, DIB);
      }
    }
    SI->eraseFromParent();
    LBI.deleteValue(SI);
  }

  // Remove dbg.assigns linked to the alloca as these are now redundant.
  at::deleteAssignmentMarkers(AI);
  AI->eraseFromParent();

  // The alloca's debuginfo can be removed as well.
  for (DbgVariableIntrinsic *DII : Info.DbgUsers)
    if (DII->isAddressOfVariable() || DII->getExpression()->startsWithDeref())
      DII->eraseFromParent();

  ++NumLocalPromoted;
  return true;
}

void PromoteMem2Reg::run() {
  Function &F = *DT.getRoot()->getParent();

  AllocaDbgUsers.resize(Allocas.size());
  AllocaATInfo.resize(Allocas.size());

  AllocaInfo Info;
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

    if (AI->use_empty()) {
      // If there are no uses of the alloca, just delete it now.
      AI->eraseFromParent();

      // Remove the alloca from the Allocas list, since it has been processed
      RemoveFromAllocasList(AllocaNum);
      ++NumDeadAlloca;
      continue;
    }

    // Calculate the set of read and write-locations for each alloca.  This is
    // analogous to finding the 'uses' and 'definitions' of each variable.
    if (Summary >= 0)
      Info = Summaries[Summary].Info;
    else
      Info.AnalyzeAlloca(AI);

    // If there is only a single store to this value, replace any loads of
    // it that are directly dominated by the definition with the value stored.
    if (Info.DefiningBlocks.size() == 1) {
      if (rewriteSingleStoreAlloca(AI, Info, LBI, SQ.DL, DT, AC)) {
        // The alloca has been processed, move on.
        RemoveFromAllocasList(AllocaNum);
        ++NumSingleStore;
        continue;
      }
    }

    // If the alloca is only read and written in one basic block, just perform a
    // linear sweep over the block to eliminate it.
    if (Info.OnlyUsedInOneBlock &&
        promoteSingleBlockAlloca(AI, Info, LBI, SQ.DL, DT, AC)) {
      // The alloca has been processed, move on.
      RemoveFromAllocasList(AllocaNum);
      continue;
    }

    // If we haven't computed a numbering for the BB's in the function, do so
    // now.
    if (BBNumbers.empty()) {
      unsigned ID = 0;
      for (auto &BB : F)
        BBNumbers[&BB] = ID++;
    }

    // Remember the dbg.declare intrinsic describing this alloca, if any.
    if (!Info.DbgUsers.empty())
      AllocaDbgUsers[AllocaNum] = Info.DbgUsers;
    if (!Info.AssignmentTracking.empty())
      AllocaATInfo[AllocaNum] = Info.AssignmentTracking;

    // Keep the reverse mapping of the 'Allocas' array for the rename pass.
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

  LBI.clear();

  // Set the incoming values for the basic block to be null values for all of
  // the alloca's.  We do this in case there is a load of a value that has not
  // been stored yet.  In this case, it will get this null value.
  RenamePassData::ValVector Values(Allocas.size());
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i)
    Values[i] = UndefValue::get(Allocas[i]->getAllocatedType());

  // When handling debug info, treat all incoming values as if they have unknown
  // locations until proven otherwise.
  RenamePassData::LocationVector Locations(Allocas.size());

  // Walks all basic blocks in the function performing the SSA rename algorithm
  // and inserting the phi nodes we marked as necessary
  std::vector<RenamePassData> RenamePassWorkList;
  RenamePassWorkList.emplace_back(&F.front(), nullptr, std::move(Values),
                                  std::move(Locations));
  do {
    RenamePassData RPD = std::move(RenamePassWorkList.back());
    RenamePassWorkList.pop_back();
    // RenamePass may add new worklist entries.
    RenamePass(RPD.BB, RPD.Pred, RPD.Values, RPD.Locations, RenamePassWorkList);
  } while (!RenamePassWorkList.empty());

  // The renamer uses the Visited set to avoid infinite loops.  Clear it now.
  Visited.clear();

  // Remove the allocas themselves from the function.
  for (Instruction *A : Allocas) {
    // Remove dbg.assigns linked to the alloca as these are now redundant.
    at::deleteAssignmentMarkers(A);
    // If there are any uses of the alloca instructions left, they must be in
    // unreachable basic blocks that were not processed by walking the dominator
    // tree. Just delete the users now.
    if (!A->use_empty())
      A->replaceAllUsesWith(PoisonValue::get(A->getType()));
    A->eraseFromParent();
  }

  // Remove alloca's dbg.declare intrinsics from the function.
  for (auto &DbgUsers : AllocaDbgUsers) {
    for (auto *DII : DbgUsers)
      if (DII->isAddressOfVariable() || DII->getExpression()->startsWithDeref())
        DII->eraseFromParent();
  }

  // Loop over all of the PHI nodes and see if there are any that we can get
  // rid of because they merge all of the same incoming values.  This can
  // happen due to undef values coming into the PHI nodes.  This process is
  // iterative, because eliminating one PHI node can cause others to be removed.
  bool EliminatedAPHI = true;
  while (EliminatedAPHI) {
    EliminatedAPHI = false;

    // Iterating over NewPhiNodes is deterministic, so it is safe to try to
    // simplify and RAUW them as we go.  If it was not, we could add uses to
    // the values we replace with in a non-deterministic order, thus creating
    // non-deterministic def->use chains.
    for (DenseMap<std::pair<unsigned, unsigned>, PHINode *>::iterator
             I = NewPhiNodes.begin(),
             E = NewPhiNodes.end();
         I != E;) {
      PHINode *PN = I->second;

      // If this PHI node merges one value and/or undefs, get the value.
      if (Value *V = simplifyInstruction(PN, SQ)) {
        PN->replaceAllUsesWith(V);
        PN->eraseFromParent();
        NewPhiNodes.erase(I++);
        EliminatedAPHI = true;
        continue;
      }
      ++I;
    }
  }

  // At this point, the renamer has added entries to PHI nodes for all reachable
  // code.  Unfortunately, there may be unreachable blocks which the renamer
  // hasn't traversed.  If this is the case, the PHI nodes may not
  // have incoming values for all predecessors.  Loop over all PHI nodes we have
  // created, inserting undef values if they are missing any incoming values.
  for (DenseMap<std::pair<unsigned, unsigned>, PHINode *>::iterator
           I = NewPhiNodes.begin(),
           E = NewPhiNodes.end();
       I != E; ++I) {
    // We want to do this once per basic block.  As such, only process a block
    // when we find the PHI that is the first entry in the block.
    PHINode *SomePHI = I->second;
    BasicBlock *BB = SomePHI->getParent();
    if (&BB->front() != SomePHI)
      continue;

    // Only do work here if there the PHI nodes are missing incoming values.  We
    // know that all PHI nodes that were inserted in a block will have the same
    // number of incoming values, so we can just check any of them.
    if (SomePHI->getNumIncomingValues() == getNumPreds(BB))
      continue;

    // Get the preds for BB.
    SmallVector<BasicBlock *, 16> Preds(predecessors(BB));

    // Ok, now we know that all of the PHI nodes are missing entries for some
    // basic blocks.  Start by sorting the incoming predecessors for efficient
    // access.
    auto CompareBBNumbers = [this](BasicBlock *A, BasicBlock *B) {
      return BBNumbers.find(A)->second < BBNumbers.find(B)->second;
    };
    llvm::sort(Preds, CompareBBNumbers);

    // Now we loop through all BB's which have entries in SomePHI and remove
    // them from the Preds list.
    for (unsigned i = 0, e = SomePHI->getNumIncomingValues(); i != e; ++i) {
      // Do a log(n) search of the Preds list for the entry we want.
      SmallVectorImpl<BasicBlock *>::iterator EntIt = llvm::lower_bound(
          Preds, SomePHI->getIncomingBlock(i), CompareBBNumbers);
      assert(EntIt != Preds.end() && *EntIt == SomePHI->getIncomingBlock(i) &&
             "PHI node has entry for a block which is not a predecessor!");

      // Remove the entry
      Preds.erase(EntIt);
    }

    // At this point, the blocks left in the preds list must have dummy
    // entries inserted into every PHI nodes for the block.  Update all the phi
    // nodes in this block that we are inserting (there could be phis before
    // mem2reg runs).
    unsigned NumBadPreds = SomePHI->getNumIncomingValues();
    BasicBlock::iterator BBI = BB->begin();
    while ((SomePHI = dyn_cast<PHINode>(BBI++)) &&
           SomePHI->getNumIncomingValues() == NumBadPreds) {
      Value *UndefVal = UndefValue::get(SomePHI->getType());
      for (BasicBlock *Pred : Preds)
        SomePHI->addIncoming(UndefVal, Pred);
    }
  }

  NewPhiNodes.clear();
}

void PromoteMem2Reg::summarizeHeavyAllocas(Function &F) {
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

/// Determine which blocks the value is live in.
///
/// These are blocks which lead to uses.  Knowing this allows us to avoid
/// inserting PHI nodes into blocks which don't lead to uses (thus, the
/// inserted phi nodes would be dead).
void PromoteMem2Reg::ComputeLiveInBlocks(
    AllocaInst *AI, AllocaInfo &Info,
    const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
    SmallPtrSetImpl<BasicBlock *> &LiveInBlocks) {
  // To determine liveness, we must iterate through the predecessors of blocks
  // where the def is live.  Blocks are added to the worklist if we need to
  // check their predecessors.  Start with all the using blocks.
  SmallVector<BasicBlock *, 64> LiveInBlockWorklist(Info.UsingBlocks.begin(),
                                                    Info.UsingBlocks.end());

  // If any of the using blocks is also a definition block, check to see if the
  // definition occurs before or after the use.  If it happens before the use,
  // the value isn't really live-in.
  for (unsigned i = 0, e = LiveInBlockWorklist.size(); i != e; ++i) {
    BasicBlock *BB = LiveInBlockWorklist[i];
    if (!DefBlocks.count(BB))
      continue;

    // Okay, this is a block that both uses and defines the value.  If the first
    // reference to the alloca is a def (store), then we know it isn't live-in.
    for (BasicBlock::iterator I = BB->begin();; ++I) {
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getOperand(1) != AI)
          continue;

        // We found a store to the alloca before a load.  The alloca is not
        // actually live-in here.
        LiveInBlockWorklist[i] = LiveInBlockWorklist.back();
        LiveInBlockWorklist.pop_back();
        --i;
        --e;
        break;
      }

      if (LoadInst *LI = dyn_cast<LoadInst>(I))
        // Okay, we found a load before a store to the alloca.  It is actually
        // live into this block.
        if (LI->getOperand(0) == AI)
          break;
    }
  }

  // Now that we have a set of blocks where the phi is live-in, recursively add
  // their predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    BasicBlock *BB = LiveInBlockWorklist.pop_back_val();

    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB).second)
      continue;

    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.  Add the preds to the worklist unless they are a
    // defining block.
    for (BasicBlock *P : predecessors(BB)) {
      // The value is not live into a predecessor if it defines the value.
      if (DefBlocks.count(P))
        continue;

      // Otherwise it is, add to the worklist.
      LiveInBlockWorklist.push_back(P);
    }
  }
}

/// Queue a phi-node to be added to a basic-block for a specific Alloca.
///
/// Returns true if there wasn't already a phi-node for that variable
bool PromoteMem2Reg::QueuePhiNode(BasicBlock *BB, unsigned AllocaNo,
                                  unsigned &Version) {
  // Look up the basic-block in question.
  PHINode *&PN = NewPhiNodes[std::make_pair(BBNumbers[BB], AllocaNo)];

  // If the BB already has a phi node added for the i'th alloca then we're done!
  if (PN)
    return false;

  // Create a PhiNode using the dereferenced type... and add the phi-node to the
  // BasicBlock.
  PN = PHINode::Create(Allocas[AllocaNo]->getAllocatedType(), getNumPreds(BB),
                       Allocas[AllocaNo]->getName() + "." + Twine(Version++),
                       &BB->front());

  if (EnableStrictDebugInformationPreservationStyle) {
    // Keep the debug information from the original alloca, to avoid discarding
    // debug information.
    PN->setDebugLoc(Allocas[AllocaNo]->getDebugLoc());
  }

  ++NumPHIInsert;
  PhiToAllocaMap[PN] = AllocaNo;
  return true;
}

/// Update the debug location of a phi. \p ApplyMergedLoc indicates whether to
/// create a merged location incorporating \p DL, or to set \p DL directly.
static void updateForIncomingValueLocation(PHINode *PN, DebugLoc DL,
                                           bool ApplyMergedLoc) {
  if (EnableStrictDebugInformationPreservationStyle and not DL) {
    // Do not override valid debug information with an empty location.
    return;
  }

  if (ApplyMergedLoc)
    PN->applyMergedLocation(PN->getDebugLoc(), DL);
  else
    PN->setDebugLoc(DL);
}

/// Recursively traverse the CFG of the function, renaming loads and
/// stores to the allocas which we are promoting.
///
/// IncomingVals indicates what value each Alloca contains on exit from the
/// predecessor block Pred.
void PromoteMem2Reg::RenamePass(BasicBlock *BB, BasicBlock *Pred,
                                RenamePassData::ValVector &IncomingVals,
                                RenamePassData::LocationVector &IncomingLocs,
                                std::vector<RenamePassData> &Worklist) {
NextIteration:
  // If we are inserting any phi nodes into this BB, they will already be in the
  // block.
  if (PHINode *APN = dyn_cast<PHINode>(BB->begin())) {
    // If we have PHI nodes to update, compute the number of edges from Pred to
    // BB.
    if (PhiToAllocaMap.count(APN)) {
      // We want to be able to distinguish between PHI nodes being inserted by
      // this invocation of mem2reg from those phi nodes that already existed in
      // the IR before mem2reg was run.  We determine that APN is being inserted
      // because it is missing incoming edges.  All other PHI nodes being
      // inserted by this pass of mem2reg will have the same number of incoming
      // operands so far.  Remember this count.
      unsigned NewPHINumOperands = APN->getNumOperands();

      unsigned NumEdges = llvm::count(successors(Pred), BB);
      assert(NumEdges && "Must be at least one edge from Pred to BB!");

      // Add entries for all the phis.
      BasicBlock::iterator PNI = BB->begin();
      do {
        unsigned AllocaNo = PhiToAllocaMap[APN];

        // Update the location of the phi node.
        updateForIncomingValueLocation(APN, IncomingLocs[AllocaNo],
                                       APN->getNumIncomingValues() > 0);

        // Add N incoming values to the PHI node.
        for (unsigned i = 0; i != NumEdges; ++i)
          APN->addIncoming(IncomingVals[AllocaNo], Pred);

        // The currently active variable for this block is now the PHI.
        IncomingVals[AllocaNo] = APN;
        AllocaATInfo[AllocaNo].updateForNewPhi(APN, DIB);
        for (DbgVariableIntrinsic *DII : AllocaDbgUsers[AllocaNo])
          if (DII->isAddressOfVariable())
            ConvertDebugDeclareToDebugValue(DII, APN, DIB);

        // Get the next phi node.
        ++PNI;
        APN = dyn_cast<PHINode>(PNI);
        if (!APN)
          break;

        // Verify that it is missing entries.  If not, it is not being inserted
        // by this mem2reg invocation so we want to ignore it.
      } while (APN->getNumOperands() == NewPHINumOperands);
    }
  }

  // Don't revisit blocks.
  if (!Visited.insert(BB).second)
    return;

  for (BasicBlock::iterator II = BB->begin(); !II->isTerminator();) {
    Instruction *I = &*II++; // get the instruction, increment iterator

    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      AllocaInst *Src = dyn_cast<AllocaInst>(LI->getPointerOperand());
      if (!Src)
        continue;

      DenseMap<AllocaInst *, unsigned>::iterator AI = AllocaLookup.find(Src);
      if (AI == AllocaLookup.end())
        continue;

      Value *V = IncomingVals[AI->second];
      convertMetadataToAssumes(LI, V, SQ.DL, AC, &DT);

      // Anything using the load now uses the current value.
      LI->replaceAllUsesWith(V);
      LI->eraseFromParent();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      // Delete this instruction and mark the name as the current holder of the
      // value
      AllocaInst *Dest = dyn_cast<AllocaInst>(SI->getPointerOperand());
      if (!Dest)
        continue;

      DenseMap<AllocaInst *, unsigned>::iterator ai = AllocaLookup.find(Dest);
      if (ai == AllocaLookup.end())
        continue;

      // what value were we writing?
      unsigned AllocaNo = ai->second;
      IncomingVals[AllocaNo] = SI->getOperand(0);

      // Record debuginfo for the store before removing it.
      IncomingLocs[AllocaNo] = SI->getDebugLoc();
      AllocaATInfo[AllocaNo].updateForDeletedStore(SI, DIB);
      for (DbgVariableIntrinsic *DII : AllocaDbgUsers[ai->second])
        if (DII->isAddressOfVariable())
          ConvertDebugDeclareToDebugValue(DII, SI, DIB);
      SI->eraseFromParent();
    }
  }

  // 'Recurse' to our successors.
  succ_iterator I = succ_begin(BB), E = succ_end(BB);
  if (I == E)
    return;

  // Keep track of the successors so we don't visit the same successor twice
  SmallPtrSet<BasicBlock *, 8> VisitedSuccs;

  // Handle the first successor without using the worklist.
  VisitedSuccs.insert(*I);
  Pred = BB;
  BB = *I;
  ++I;

  for (; I != E; ++I)
    if (VisitedSuccs.insert(*I).second)
      Worklist.emplace_back(*I, Pred, IncomingVals, IncomingLocs);

  goto NextIteration;
}

void llvm::PromoteMemToReg(ArrayRef<AllocaInst *> Allocas, DominatorTree &DT,
                           AssumptionCache *AC) {
  // If there is nothing to do, bail out...
  if (Allocas.empty())
    return;

  PromoteMem2Reg(Allocas, DT, AC).run();
}
