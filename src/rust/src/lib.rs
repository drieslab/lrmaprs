//! Compiled kernels for lrmap: the loops over toroidal shifts of the sender layer.
//!
//! * `rs_contact_scores`: per shift, the k nearest (shifted) sender cells within `cap` of each receiver (spatial
//!   hash), weights exp(-d / d0) normalised per receiver, and for every test (sender type A, receiver type B,
//!   ligand column, receptor column) the score sum_i R_i sum_j wn_ij L_j / n_valid(B), plus the sampling
//!   variance of that score for the first `nvar` shifts.
//! * `rs_grid_gather`: per shift, sum_i R_i F(bin of receiver i minus the shift) and its receptor-side variance.
//! * `rs_blur`: separable Gaussian blur of many grids (zero padded).
//! Shifts run in parallel when `threads` > 1; `threads` = 1 runs on the calling thread.

use extendr_api::prelude::*;
use extendr_api::Result;
use rayon::prelude::*;

fn ints(x: &Robj, what: &str) -> Result<Vec<i32>> {
    x.as_integer_slice().map(|s| s.to_vec()).ok_or_else(|| Error::Other(format!("{what} must be an integer vector")))
}
fn reals<'a>(x: &'a Robj, what: &str) -> Result<&'a [f64]> {
    x.as_real_slice().ok_or_else(|| Error::Other(format!("{what} must be a double vector or matrix")))
}
fn dims(x: &Robj) -> (usize, usize) {
    match x.get_attrib("dim").and_then(|d| d.as_integer_slice().map(|s| s.to_vec())) {
        Some(v) if v.len() >= 2 => (v[0] as usize, v[1] as usize),
        _ => (x.len(), 1),
    }
}
fn blocks<'a>(m: &'a [Robj], v: &'a [Robj]) -> Vec<Block<'a>> {
    m.iter().zip(v.iter()).map(|(a, b)| Block { m: a.as_real_slice().unwrap_or(&[]), v: b.as_real_slice().unwrap_or(&[]),
                                                nr: dims(a).0 }).collect()
}
fn to_matrix(v: &[f64], nr: usize, nc: usize) -> RMatrix<f64> {
    let mut m = RMatrix::<f64>::new(nr, nc);
    m.data_mut().copy_from_slice(v);
    m
}
fn run<T: Send, F: Fn(usize) -> T + Sync + Send>(n: usize, threads: i32, f: F) -> Vec<T> {
    if threads == 1 {
        (0..n).map(f).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(0) as usize).build().unwrap();
        pool.install(|| (0..n).into_par_iter().map(f).collect())
    }
}

struct Block<'a> {
    m: &'a [f64],
    v: &'a [f64],
    nr: usize,
}

/// Contact-scale scores of lrmap over all toroidal shifts
/// @description Internal kernel of lrmap (called by `lrmap(backend = "rust")`). For every shift of the sender layer it
///   finds each receiver's `k` nearest sender-layer cells within `cap` (spatial hash), weights them exp(-d / d0),
///   and sums R_i * w_ij * L_j per test, plus the sampling-variance sums of the first `nvar` shifts.
/// @param x,y cell coordinates (cells sorted by type). @param tcode cell type codes (1-based, sorted).
/// @param off 0-based offset of each type's first cell. @param lb,lv per sender type: dense ligand values and
///   their variances (cells x entities; empty matrix if unused). @param rb,rv the same for receptors per receiver type.
/// @param ta,tb sender and receiver type of each test. @param tlc,trc ligand and receptor column of each test in
///   the type's block. @param shifts shifts x 2 matrix (microns; first row the observed layout).
/// @param k neighbours. @param d0 distance scale of the weights. @param cap maximum neighbour distance.
/// @param bx bounding box (xmin, width, ymin, height) for the toroidal wrap. @param nvar shifts with variance sums.
/// @param threads threads (1: calling thread, 0: all cores).
/// @return list(S = tests x shifts scores, Vs = tests x nvar variance sums).
/// @keywords internal
/// @export
#[extendr]
fn rs_contact_scores(x: Robj, y: Robj, tcode: Robj, off: Robj, lb: List, lv: List, rb: List, rv: List,
                     ta: Robj, tb: Robj, tlc: Robj, trc: Robj, shifts: Robj, k: i32, d0: f64, cap: f64,
                     bx: Robj, nvar: i32, threads: i32) -> Result<List> {
    let x = reals(&x, "x")?; let y = reals(&y, "y")?; let n = x.len();
    let tc: Vec<usize> = ints(&tcode, "tcode")?.iter().map(|&t| (t - 1) as usize).collect();
    let off: Vec<usize> = ints(&off, "off")?.iter().map(|&o| o as usize).collect();
    let kk = off.len();
    let keep: Vec<Vec<Robj>> = vec![lb.values().collect(), lv.values().collect(), rb.values().collect(), rv.values().collect()];
    let lbs = blocks(&keep[0], &keep[1]); let rbs = blocks(&keep[2], &keep[3]);
    let ta = ints(&ta, "ta")?; let tb = ints(&tb, "tb")?; let tlc = ints(&tlc, "tlc")?; let trc = ints(&trc, "trc")?;
    let nt = ta.len();
    // tests per (A, B) block: (row, ligand column, receptor column), 0-based
    let mut bl: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); kk * kk];
    for r in 0..nt {
        let a = (ta[r] - 1) as usize; let b = (tb[r] - 1) as usize;
        bl[a * kk + b].push((r, (tlc[r] - 1) as usize, (trc[r] - 1) as usize));
    }
    let sh = reals(&shifts, "shifts")?; let (nsh, _) = dims(&shifts);
    let bx = reals(&bx, "box")?; let (x0, w, y0, h) = (bx[0], bx[1], bx[2], bx[3]);
    let k = k.max(1) as usize; let nvar = (nvar.max(0) as usize).min(nsh);
    let ncx = ((w / cap).ceil() as usize).max(1); let ncy = ((h / cap).ceil() as usize).max(1);
    let cell = |px: f64, py: f64| -> (usize, usize) {
        (((px - x0) / cap).floor().clamp(0.0, (ncx - 1) as f64) as usize,
         ((py - y0) / cap).floor().clamp(0.0, (ncy - 1) as f64) as usize)
    };

    let one = |v: usize| -> (Vec<f64>, Vec<f64>) {
        // sender positions for this shift (v = 0: observed layout, self excluded)
        let (dx, dy) = (sh[v], sh[nsh + v]);
        let (sx, sy): (Vec<f64>, Vec<f64>) = if v == 0 { (x.to_vec(), y.to_vec()) } else {
            ((0..n).map(|j| x0 + (x[j] - x0 + dx).rem_euclid(w)).collect(),
             (0..n).map(|j| y0 + (y[j] - y0 + dy).rem_euclid(h)).collect())
        };
        // spatial hash (counting sort by cell)
        let mut cnt = vec![0usize; ncx * ncy + 1];
        let cj: Vec<usize> = (0..n).map(|j| { let (a, b) = cell(sx[j], sy[j]); b * ncx + a }).collect();
        for &c in &cj { cnt[c + 1] += 1; }
        for c in 0..ncx * ncy { cnt[c + 1] += cnt[c]; }
        let mut pos = cnt.clone(); let mut pts = vec![0usize; n];
        for j in 0..n { pts[pos[cj[j]]] = j; pos[cj[j]] += 1; }
        // edges: k nearest within cap
        let mut er: Vec<usize> = Vec::with_capacity(n * k); let mut es: Vec<usize> = Vec::with_capacity(n * k);
        let mut ew: Vec<f64> = Vec::with_capacity(n * k);
        let mut nv = vec![0f64; kk];
        let mut cand: Vec<(f64, usize)> = Vec::new();
        for i in 0..n {
            cand.clear();
            let (ci, cy) = cell(x[i], y[i]);
            for gy in cy.saturating_sub(1)..=(cy + 1).min(ncy - 1) {
                for gx in ci.saturating_sub(1)..=(ci + 1).min(ncx - 1) {
                    let c = gy * ncx + gx;
                    for &j in &pts[cnt[c]..cnt[c + 1]] {
                        if v == 0 && j == i { continue; }
                        let d = ((sx[j] - x[i]).powi(2) + (sy[j] - y[i]).powi(2)).sqrt();
                        if d <= cap { cand.push((d, j)); }
                    }
                }
            }
            if cand.is_empty() { continue; }
            if cand.len() > k { cand.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap()); cand.truncate(k); }
            let tot: f64 = cand.iter().map(|c| (-c.0 / d0).exp()).sum();
            for c in cand.iter() { er.push(i); es.push(c.1); ew.push((-c.0 / d0).exp() / tot); }
            nv[tc[i]] += 1.0;
        }
        let mut s = vec![0f64; nt];
        for e in 0..er.len() {
            let (i, j) = (er[e], es[e]); let (a, b) = (tc[j], tc[i]);
            let rows = &bl[a * kk + b]; if rows.is_empty() { continue; }
            let (il, jl) = (i - off[b], j - off[a]); let (rbk, lbk) = (&rbs[b], &lbs[a]);
            for &(r, lc, rc) in rows {
                s[r] += ew[e] * rbk.m[rc * rbk.nr + il] * lbk.m[lc * lbk.nr + jl];
            }
        }
        let mut vs = vec![0f64; if v < nvar { nt } else { 0 }];
        if v < nvar {
            // receptor side: per receiver, X_i = sum_j wn L_j per test, then X_i^2 * var(R_i)
            let mut acc = vec![0f64; nt]; let mut touched: Vec<usize> = Vec::new();
            let mut e = 0;
            while e < er.len() {
                let i = er[e]; let b = tc[i]; let il = i - off[b];
                while e < er.len() && er[e] == i {
                    let j = es[e]; let a = tc[j]; let jl = j - off[a]; let lbk = &lbs[a];
                    for &(r, lc, _) in &bl[a * kk + b] {
                        if acc[r] == 0.0 { touched.push(r); }
                        acc[r] += ew[e] * lbk.m[lc * lbk.nr + jl];
                    }
                    e += 1;
                }
                let rbk = &rbs[b];
                for &r in &touched {
                    let rc = (trc[r] - 1) as usize;
                    vs[r] += acc[r] * acc[r] * rbk.v[rc * rbk.nr + il]; acc[r] = 0.0;
                }
                touched.clear();
            }
            // ligand side: per sender, C_j = sum_i wn R_i per test, then C_j^2 * var(L_j)
            let mut ord: Vec<usize> = (0..er.len()).collect(); ord.sort_unstable_by_key(|&e| es[e]);
            let mut p = 0;
            while p < ord.len() {
                let j = es[ord[p]]; let a = tc[j]; let jl = j - off[a];
                while p < ord.len() && es[ord[p]] == j {
                    let e = ord[p]; let i = er[e]; let b = tc[i]; let il = i - off[b]; let rbk = &rbs[b];
                    for &(r, _, rc) in &bl[a * kk + b] {
                        if acc[r] == 0.0 { touched.push(r); }
                        acc[r] += ew[e] * rbk.m[rc * rbk.nr + il];
                    }
                    p += 1;
                }
                let lbk = &lbs[a];
                for &r in &touched {
                    let lc = (tlc[r] - 1) as usize;
                    vs[r] += acc[r] * acc[r] * lbk.v[lc * lbk.nr + jl]; acc[r] = 0.0;
                }
                touched.clear();
            }
            for r in 0..nt { let b = (tb[r] - 1) as usize; vs[r] /= nv[b].max(1.0).powi(2); }
        }
        for r in 0..nt { let b = (tb[r] - 1) as usize; s[r] /= nv[b].max(1.0); }
        (s, vs)
    };
    let res = run(nsh, threads, one);
    let mut s = vec![0f64; nt * nsh]; let mut vs = vec![0f64; nt * nvar];
    for (v, (sv, vv)) in res.into_iter().enumerate() {
        s[v * nt..(v + 1) * nt].copy_from_slice(&sv);
        if v < nvar { vs[v * nt..(v + 1) * nt].copy_from_slice(&vv); }
    }
    Ok(list!(S = to_matrix(&s, nt, nsh), Vs = to_matrix(&vs, nt, nvar)))
}

/// Grid-scale scores of lrmap over all toroidal shifts
/// @description Internal kernel of lrmap. Reads blurred ligand fields at the (shifted) bins of receiver cells and
///   sums R_i * X_i per test, with receptor-side variance sums for the first `nvar` shifts.
/// @param fq bins x ligands matrix of blurred ligand shares. @param gx,gy receiver bins (0-based).
/// @param sx,sy shifts in bins. @param nx,ny grid size. @param rm,vr receivers x tests receptor values and
///   variances. @param lp column of `fq` for each test (1-based). @param nv valid receivers per shift (score
///   denominator). @param nvar shifts with variance sums. @param threads threads (1: calling thread, 0: all cores).
/// @return list(S = tests x shifts scores, Vs = tests x nvar variance sums).
/// @keywords internal
/// @export
#[extendr]
fn rs_grid_gather(fq: Robj, gx: Robj, gy: Robj, sx: Robj, sy: Robj, nx: i32, ny: i32, rm: Robj, vr: Robj,
                  lp: Robj, nv: Robj, nvar: i32, threads: i32) -> Result<List> {
    let (nc, _) = dims(&fq); let fq = reals(&fq, "fq")?;
    let gx = ints(&gx, "gx")?; let gy = ints(&gy, "gy")?; let sx = ints(&sx, "sx")?; let sy = ints(&sy, "sy")?;
    let (nb, nrow) = dims(&rm); let rm = reals(&rm, "rm")?; let vr = reals(&vr, "vr")?;
    let lp: Vec<usize> = ints(&lp, "lp")?.iter().map(|&c| (c - 1) as usize).collect();
    let nv = reals(&nv, "nv")?; let nsh = sx.len(); let nvar = (nvar.max(0) as usize).min(nsh);
    let (nx, ny) = (nx as i64, ny as i64);
    let one = |v: usize| -> (Vec<f64>, Vec<f64>) {
        let lin: Vec<usize> = (0..nb).map(|i| {
            ((gx[i] as i64 - sx[v] as i64).rem_euclid(nx) + (gy[i] as i64 - sy[v] as i64).rem_euclid(ny) * nx) as usize
        }).collect();
        let mut s = vec![0f64; nrow]; let mut vs = vec![0f64; if v < nvar { nrow } else { 0 }];
        let d = nv[v].max(1.0);
        for r in 0..nrow {
            let col = &fq[lp[r] * nc..(lp[r] + 1) * nc]; let rr = &rm[r * nb..(r + 1) * nb];
            let mut a = 0f64;
            for i in 0..nb { a += rr[i] * col[lin[i]]; }
            s[r] = a / d;
            if v < nvar {
                let vv = &vr[r * nb..(r + 1) * nb]; let mut b = 0f64;
                for i in 0..nb { let xv = col[lin[i]]; b += xv * xv * vv[i]; }
                vs[r] = b / (d * d);
            }
        }
        (s, vs)
    };
    let res = run(nsh, threads, one);
    let mut s = vec![0f64; nrow * nsh]; let mut vs = vec![0f64; nrow * nvar];
    for (v, (sv, vv)) in res.into_iter().enumerate() {
        s[v * nrow..(v + 1) * nrow].copy_from_slice(&sv);
        if v < nvar { vs[v * nrow..(v + 1) * nrow].copy_from_slice(&vv); }
    }
    Ok(list!(S = to_matrix(&s, nrow, nsh), Vs = to_matrix(&vs, nrow, nvar)))
}

/// Separable Gaussian blur of several grids
/// @description Internal kernel of lrmap: blurs each column (an nx x ny grid, column-major) with a symmetric kernel
///   along x and then y, zero padded.
/// @param v (nx * ny) x m matrix. @param nx,ny grid size. @param ker kernel weights (odd length, symmetric).
/// @param threads threads (1: calling thread, 0: all cores).
/// @return matrix of the same shape.
/// @keywords internal
/// @export
#[extendr]
fn rs_blur(v: Robj, nx: i32, ny: i32, ker: Robj, threads: i32) -> Result<RMatrix<f64>> {
    let (nc, m) = dims(&v); let v = reals(&v, "v")?; let ker = reals(&ker, "ker")?;
    let (nx, ny) = (nx as usize, ny as usize); let r = (ker.len() - 1) / 2;
    if nc != nx * ny { return Err(Error::Other("rows must be nx * ny".into())); }
    let one = |c: usize| -> Vec<f64> {
        let src = &v[c * nc..(c + 1) * nc];
        let mut tmp = vec![0f64; nc];
        for yy in 0..ny {
            let row = &src[yy * nx..(yy + 1) * nx]; let out = &mut tmp[yy * nx..(yy + 1) * nx];
            for (xx, val) in row.iter().enumerate() {
                if *val == 0.0 { continue; }
                let lo = xx.saturating_sub(r); let hi = (xx + r).min(nx - 1);
                for t in lo..=hi { out[t] += ker[t + r - xx] * val; }
            }
        }
        let mut out = vec![0f64; nc];
        for yy in 0..ny {
            let lo = yy.saturating_sub(r); let hi = (yy + r).min(ny - 1);
            for t in lo..=hi {
                let kv = ker[t + r - yy];
                let (s, d) = (&tmp[yy * nx..(yy + 1) * nx], t * nx);
                for xx in 0..nx { out[d + xx] += kv * s[xx]; }
            }
        }
        out
    };
    let res = run(m, threads, one);
    let mut out = RMatrix::<f64>::new(nc, m);
    { let d = out.data_mut(); for (c, col) in res.into_iter().enumerate() { d[c * nc..(c + 1) * nc].copy_from_slice(&col); } }
    Ok(out)
}

extendr_module! {
    mod lrmaprs;
    fn rs_contact_scores;
    fn rs_grid_gather;
    fn rs_blur;
}
