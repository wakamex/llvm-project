# Patches the rev.ng fork's runtimes were built with

The LLVM libraries of the [rev.ng fork](https://github.com/wakamex/revng)'s runtimes were built from rev.ng's LLVM commit `b93d654bc9267ca2128528583558db618ff9cc1d` with these patches. Each is also a commit on this repository's branches, which is the form to read and build from; these files are the exact sources, kept for provenance.

| File | SHA-256 | Library | Commit |
| --- | --- | --- | --- |
| `llvm-handle-hash.patch` | `c23946d5577cc547...` | libLLVMCore | `bcbe0cd9de6` on `fix/large-module-scaling` |
| `llvm-blockaddress-index.patch` | `560276e934964d0e...` | libLLVMBitReader | `789d9761a6c` on `fix/large-module-scaling` |
| `jump-threading-change.patch` | `3e8ff2614cbf7cd6...` | libLLVMAnalysis, libLLVMTransformUtils | `9c6832e4a3b` on `fix/large-module-scaling` |
| `batch_patch2.py` | `4fc8ab56e11025f2...` | libLLVMTransformUtils | `f04c9134ef0` on `fix/large-module-scaling` |
| `llvm-lvi-cache-contents.patch` | `daf1b86bc508ed4a...` | libLLVMAnalysis | `f30695bb4c6` on `diag/lvi-cache-counters` |

- `jump-threading-change.patch` was lost with its work directory and replayed from the two edit scripts recorded when it was written (`jump-threading-edit-1.py`, `jump-threading-edit-2.py`, applied in order to the pristine `Local.cpp` and `DomTreeUpdater.cpp`). Its added lines match the original; its SHA-256 differs from the recorded `d74291bb...` only through the file timestamps `diff -ru` writes into the headers. Both changed functions compile to the same machine code as the shipped libraries.
- `batch_patch2.py` is the mem2reg change as the script that produced it: run on `llvm/lib/Transforms/Utils/PromoteMemoryToRegister.cpp` at the base commit, `python3 batch_patch2.py PromoteMemoryToRegister.cpp out.cpp` writes the patched file, SHA-256 `91b1597381c9341e29853342f656f4beb73bbf6dd083ec2d5cf306d94aa5f645`, the source the runtimes were built from.
- `llvm-lvi-cache-contents.patch` is instrumentation the fork's trace uses through weak references; it is not proposed for upstream.
