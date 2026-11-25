test_that("rs_blur equals a direct separable convolution (zero padded), with 1 and 2 threads", {
  set.seed(1); nx <- 13L; ny <- 9L; m <- 3
  V <- matrix(as.numeric(stats::rpois(nx * ny * m, 0.5)), nx * ny, m)
  ker <- exp(-0.5 * ((-3):3)^2 / 2)
  ref <- vapply(seq_len(m), function(c) {
    G <- matrix(V[, c], nx, ny); out <- matrix(0, nx, ny)
    for (i in seq_len(nx)) for (j in seq_len(ny)) for (a in -3:3) for (b in -3:3) {
      ii <- i + a; jj <- j + b
      if (ii >= 1 && ii <= nx && jj >= 1 && jj <= ny) out[i, j] <- out[i, j] + ker[a + 4] * ker[b + 4] * G[ii, jj]
    }
    as.vector(out)
  }, numeric(nx * ny))
  expect_equal(rs_blur(V, nx, ny, ker, 1L), ref, tolerance = 1e-12)
  expect_identical(rs_blur(V, nx, ny, ker, 2L), rs_blur(V, nx, ny, ker, 1L))
})

test_that("lrmap gives the same results with this backend and in R", {
  skip_if_not_installed("lrmap")
  s <- lrmap::lrmap_simulate("touch", size = 500, n_clusters = 3, seed = 2)
  run <- function(b) suppressWarnings(lrmap::lrmap(s$counts, s$coords, s$types, s$lr, lib = s$lib, B = 10,
                                                   min_shift = 150, backend = b, verbose = FALSE))
  expect_equal(run("rust")$table, run("r")$table, tolerance = 1e-9)
})
