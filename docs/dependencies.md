# Dependency and environment record

Observed locally: macOS 15.7 (24G222), Darwin 24.6.0, arm64; Apple Clang
17.0.0 (clang-1700.6.4.2), target arm64-apple-darwin24.6.0; Xcode toolchain
at `/Applications/Xcode.app/Contents/Developer`. Python 3.11.9 from pyenv,
CMake 3.31.6 in the project virtual environment. The default system Python
was 3.14.2 and was not modified. Initial directory was empty with no Git
repository or AGENTS.md; this independent project occupies that directory.

Fetched C++ dependencies (no moving main branches):

| Dependency | Release archive | SHA-256 |
|---|---|---|
| Eigen 3.4.0 | `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz` | `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72` |
| GoogleTest 1.17.0 | `https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz` | `65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c` |

Release references: [Eigen 3.4.0 announcement](https://www.eigen.tuxfamily.org/index.php?title=News%3AEigen_3.4.0_released%21),
[GoogleTest 1.17.0](https://github.com/google/googletest/releases/tag/v1.17.0).
Eigen and GoogleTest retain their own licenses in the fetched sources.

Direct Python requirements are in `requirements.txt`; exact resolved versions,
including transitive packages, are in `requirements-lock.txt`. The demo also
records `pip freeze`. Main versions: NumPy 2.2.6, matplotlib 3.10.3, pytest 8.3.5.
No global package installations were performed. Initial sandboxed network
access failed; approved downloads into `.venv` and `.deps` succeeded.

Project targets compile as C++17 with `-Wall -Wextra -Wpedantic -Wconversion
-Wshadow -Werror`; Debug builds use the compiler's debug flags. Optional
sanitizers add `-fsanitize=address,undefined -fno-omit-frame-pointer` at compile
time and both sanitizers at link time. Third-party headers are system includes.
