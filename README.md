# lrmaprs

The Rust backend for [lrmap](https://github.com/drieslab/lrmap). lrmap is pure R and works without this package. With it installed, `lrmap(backend = "auto")` runs its inner loops here:

- `rs_contact_scores()`: contact-scale scores for every toroidal shift. A spatial hash finds each receiver's nearest cells in the shifted sender layer, and sums are kept per test.
- `rs_grid_gather()`: paracrine- and niche-scale scores for every shift, read from blurred ligand fields.
- `rs_blur()`: separable Gaussian blur of several grids at once.

Results are identical to lrmap's R code (max |Δz| about 4e-14 on a breast section). On a 22k-cell crop with 1,387 ligand-receptor pairs, a full run took 93 s in R and 30 s with this package on one thread (21 s on two).

`threads = 1` runs on the calling thread; `threads = 0` uses all cores.

## Build
- **Requirements:** Cargo and rustc (≥ 1.65).
- **Crates:** the build downloads them; a `src/rust/vendor.tar.xz` (from `rextendr::vendor_pkgs()`) is used offline when present.
- **Bindings:** extendr 0.9.
