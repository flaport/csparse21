#![crate_name = "csparse21"]

//! Solving large systems of linear equations using sparse matrix methods.
//!
//! [![Docs](https://docs.rs/csparse21/badge.svg)](docs.rs/csparse21)
//!
//! ```rust
//! let mut m = csparse21::Matrix::from_entries(vec![
//!             (0, 0, 1.0),
//!             (0, 1, 1.0),
//!             (0, 2, 1.0),
//!             (1, 1, 2.0),
//!             (1, 2, 5.0),
//!             (2, 0, 2.0),
//!             (2, 1, 5.0),
//!             (2, 2, -1.0),
//!         ]);
//!
//! let soln = m.solve(vec![6.0, -4.0, 27.0]);
//! // => vec![5.0, 3.0, -2.0]
//! ```
//!
//! Sparse methods are primarily valuable for systems in which the number of non-zero entries is substantially less than the overall size of the matrix. Such situations are common in physical systems, including electronic circuit simulation. All elements of a sparse matrix are assumed to be zero-valued unless indicated otherwise.
//!
//! ## Usage
//!
//! CSparse21 exposes two primary data structures:
//!
//! * `Matrix` represents an `Complex64`-valued sparse matrix
//! * `System` represents a system of linear equations of the form `Ax=b`, including a `Matrix` (A) and right-hand-side `Vec` (b).
//!
//! Once matrices and systems have been created, their primary public method is `solve`, which returns a (dense) `Vec` solution-vector.
//!

use num_complex::Complex64;
use std::cmp::{max, min};
use std::error::Error;
use std::fmt;
use std::ops::{Index, IndexMut};
use std::usize::MAX;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Eindex(usize);

// `Entry`s are a type alias for tuples of (row, col, val).
type Entry = (usize, usize, Complex64);

#[derive(Debug, Copy, Clone)]
enum Axis {
    ROWS = 0,
    COLS,
}

use Axis::*;

impl Axis {
    fn other(&self) -> Axis {
        match self {
            Axis::ROWS => Axis::COLS,
            Axis::COLS => Axis::ROWS,
        }
    }
}

struct AxisPair<T> {
    rows: T,
    cols: T,
}

impl<T> Index<Axis> for AxisPair<T> {
    type Output = T;

    fn index(&self, ax: Axis) -> &Self::Output {
        match ax {
            Axis::ROWS => &self.rows,
            Axis::COLS => &self.cols,
        }
    }
}

impl<T> IndexMut<Axis> for AxisPair<T> {
    fn index_mut(&mut self, ax: Axis) -> &mut Self::Output {
        match ax {
            Axis::ROWS => &mut self.rows,
            Axis::COLS => &mut self.cols,
        }
    }
}

#[derive(PartialEq, Debug, Copy, Clone)]
enum MatrixState {
    CREATED = 0,
    FACTORING,
    FACTORED,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Element {
    index: Eindex,
    row: usize,
    col: usize,
    val: Complex64,
    fillin: bool,
    orig: (usize, usize, Complex64),
    next_in_row: Option<Eindex>,
    next_in_col: Option<Eindex>,
}

impl PartialEq for Element {
    fn eq(&self, other: &Self) -> bool {
        return self.row == other.row && self.col == other.col && self.val == other.val;
    }
}

impl Element {
    fn new(index: Eindex, row: usize, col: usize, val: Complex64, fillin: bool) -> Element {
        Element {
            index,
            row,
            col,
            val,
            fillin,
            orig: (row, col, val),
            next_in_row: None,
            next_in_col: None,
        }
    }
    fn loc(&self, ax: Axis) -> usize {
        match ax {
            Axis::ROWS => self.row,
            Axis::COLS => self.col,
        }
    }
    fn set_loc(&mut self, ax: Axis, to: usize) {
        match ax {
            Axis::ROWS => self.row = to,
            Axis::COLS => self.col = to,
        }
    }
    fn next(&self, ax: Axis) -> Option<Eindex> {
        match ax {
            Axis::ROWS => self.next_in_row,
            Axis::COLS => self.next_in_col,
        }
    }
    fn set_next(&mut self, ax: Axis, e: Option<Eindex>) {
        match ax {
            Axis::ROWS => self.next_in_row = e,
            Axis::COLS => self.next_in_col = e,
        }
    }
}

struct AxisMapping {
    e2i: Vec<usize>,
    i2e: Vec<usize>,
    history: Vec<(usize, usize)>,
}

impl AxisMapping {
    fn new(size: usize) -> AxisMapping {
        AxisMapping {
            e2i: (0..size).collect(),
            i2e: (0..size).collect(),
            history: vec![],
        }
    }
    fn swap_int(&mut self, x: usize, y: usize) {
        // Swap internal indices x and y
        let tmp = self.i2e[x];
        self.i2e[x] = self.i2e[y];
        self.i2e[y] = tmp;
        self.e2i[self.i2e[x]] = x;
        self.e2i[self.i2e[y]] = y;
        self.history.push((x, y));
    }
}

#[allow(dead_code)]
struct AxisData {
    ax: Axis,
    hdrs: Vec<Option<Eindex>>,
    qtys: Vec<usize>,
    markowitz: Vec<usize>,
    mapping: Option<AxisMapping>,
}

impl AxisData {
    fn new(ax: Axis) -> AxisData {
        AxisData {
            ax,
            hdrs: vec![],
            qtys: vec![],
            markowitz: vec![],
            mapping: None,
        }
    }
    fn grow(&mut self, to: usize) {
        if to <= self.hdrs.len() {
            return;
        }
        let by = to - self.hdrs.len();
        for _ in 0..by {
            self.hdrs.push(None);
            self.qtys.push(0);
            self.markowitz.push(0);
        }
    }
    fn setup_factoring(&mut self) {
        self.markowitz.copy_from_slice(&self.qtys);
        self.mapping = Some(AxisMapping::new(self.hdrs.len()));
    }
    fn swap(&mut self, x: usize, y: usize) {
        self.hdrs.swap(x, y);
        self.qtys.swap(x, y);
        self.markowitz.swap(x, y);
        if let Some(m) = &mut self.mapping {
            m.swap_int(x, y);
        }
    }
}

type SpResult<T> = Result<T, &'static str>;

/// Sparse Matrix
pub struct Matrix {
    // Matrix.elements is the owner of all `Element`s.
    // Everything else gets referenced via `Eindex`es.
    state: MatrixState,
    elements: Vec<Element>,
    axes: AxisPair<AxisData>,
    diag: Vec<Option<Eindex>>,
    fillins: Vec<Eindex>,
}

impl Matrix {
    /// Create a new, initially empty `Matrix`
    pub fn new() -> Matrix {
        Matrix {
            state: MatrixState::CREATED,
            axes: AxisPair {
                rows: AxisData::new(Axis::ROWS),
                cols: AxisData::new(Axis::COLS),
            },
            diag: vec![],
            elements: vec![],
            fillins: vec![],
        }
    }
    /// Create a new `Matrix` from a vector of (row, col, val) `entries`.
    pub fn from_entries(entries: Vec<Entry>) -> Matrix {
        let mut m = Matrix::new();
        for e in entries.iter() {
            m.add_element(e.0, e.1, e.2);
        }
        return m;
    }
    /// Create an n*n identity `Matrix`
    ///
    pub fn identity(n: usize) -> Matrix {
        let one = Complex64 { re: 1.0, im: 0.0 };
        let mut m = Matrix::new();
        for k in 0..n {
            m.add_element(k, k, one);
        }
        return m;
    }
    /// Add an element at location `(row, col)` with value `val`.
    pub fn add_element(&mut self, row: usize, col: usize, val: Complex64) {
        self._add_element(row, col, val, false);
    }
    /// Add elements correspoding to each triplet `(row, col, val)`
    /// Rows and columns are `usize`, and `vals` are `Complex64`.
    pub fn add_elements(&mut self, elements: Vec<Entry>) {
        for e in elements.iter() {
            self.add_element(e.0, e.1, e.2);
        }
    }
    /// Create a zero-valued element at `(row, col)`,
    /// or return existing Element index if present
    pub fn make(&mut self, row: usize, col: usize) -> Eindex {
        let zero = Complex64 { re: 0.0, im: 0.0 };
        return match self.get_elem(row, col) {
            Some(ei) => ei,
            None => self._add_element(row, col, zero, false),
        };
    }
    /// Reset all Elements to zero value.
    pub fn reset(&mut self) {
        let zero = Complex64 { re: 0.0, im: 0.0 };
        for e in self.elements.iter_mut() {
            e.val = zero;
        }
        self.set_state(MatrixState::CREATED).unwrap();
    }
    /// Update `Element` `ei` by `val`
    pub fn update(&mut self, ei: Eindex, val: Complex64) {
        self[ei].val += val;
    }
    /// Multiply by Vec
    pub fn vecmul(&self, x: &Vec<Complex64>) -> SpResult<Vec<Complex64>> {
        if x.len() != self.num_cols() {
            return Err("Invalid Dimensions");
        }
        let zero = Complex64 { re: 0.0, im: 0.0 };
        let mut y: Vec<Complex64> = vec![zero; self.num_rows()];
        for row in 0..self.num_rows() {
            let mut ep = self.hdr(ROWS, row);
            while let Some(ei) = ep {
                y[row] += self[ei].val * x[self[ei].col];
                ep = self[ei].next_in_row;
            }
        }
        return Ok(y);
    }
    pub fn res(&self, x: &Vec<Complex64>, rhs: &Vec<Complex64>) -> SpResult<Vec<Complex64>> {
        println!("X:");
        println!("{:?}", x);

        let zero = Complex64 { re: 0.0, im: 0.0 };
        let mut xi: Vec<Complex64> = vec![zero; self.num_cols()];
        if self.state == MatrixState::FACTORED {
            // If we have factored, unwind any column-swaps
            let col_mapping = self.axes[COLS].mapping.as_ref().unwrap();
            let row_mapping = self.axes[ROWS].mapping.as_ref().unwrap();
            println!("COL_MAP:");
            println!("{:?}", col_mapping.e2i);
            println!("ROW_MAP:");
            println!("{:?}", row_mapping.e2i);
            for k in 0..xi.len() {
                xi[k] = x[col_mapping.e2i[k]];
            }
        } else {
            for k in 0..xi.len() {
                xi[k] = x[k];
            }
        }

        println!("XI:");
        println!("{:?}", xi);

        let m: Vec<Complex64> = self.vecmul(&xi)?;
        let zero = Complex64 { re: 0.0, im: 0.0 };
        let mut res = vec![zero; m.len()];

        if self.state == MatrixState::FACTORED {
            let row_mapping = self.axes[ROWS].mapping.as_ref().unwrap();
            for k in 0..xi.len() {
                res[k] = rhs[row_mapping.e2i[k]] - m[k];
            }
        } else {
            for k in 0..xi.len() {
                res[k] = rhs[k] - m[k];
            }
        }
        // for k in 0..self.x.len() {
        //     res[k] = -1.0 * res[k];
        //     // res[k] += self.rhs[k];
        // }
        // println!("RES_BEFORE_RHS:");
        // println!("{:?}", res);
        // for k in 0..self.x.len() {
        //     res[k] += self.rhs[k];
        // }
        // println!("RES_WITH_RHS:");
        // println!("{:?}", res);
        // // res[0] += self.rhs[0];
        // // res[1] += self.rhs[2];
        // // res[2] += self.rhs[1];
        return Ok(res);
    }
    fn insert(&mut self, e: &mut Element) {
        let mut expanded = false;
        if e.row + 1 > self.num_rows() {
            self.axes[Axis::ROWS].grow(e.row + 1);
            expanded = true;
        }
        if e.col + 1 > self.num_cols() {
            self.axes[Axis::COLS].grow(e.col + 1);
            expanded = true;
        }
        if expanded {
            let new_diag_len = std::cmp::min(self.num_rows(), self.num_cols());
            for _ in 0..new_diag_len - self.diag.len() {
                self.diag.push(None);
            }
        }
        // Insert along each Axis
        self.insert_axis(Axis::COLS, e);
        self.insert_axis(Axis::ROWS, e);
        // Update row & col qtys
        self.axes[ROWS].qtys[e.row] += 1;
        self.axes[COLS].qtys[e.col] += 1;
        if self.state == MatrixState::FACTORING {
            self.axes[ROWS].markowitz[e.row] += 1;
            self.axes[COLS].markowitz[e.col] += 1;
        }
        // Update our special arrays
        if e.row == e.col {
            self.diag[e.row] = Some(e.index);
        }
        if e.fillin {
            self.fillins.push(e.index);
        }
    }
    fn insert_axis(&mut self, ax: Axis, e: &mut Element) {
        // Insert Element `e` along Axis `ax`

        let head_ptr = self.axes[ax].hdrs[e.loc(ax)];
        let head_idx = match head_ptr {
            Some(h) => h,
            None => {
                // Adding first element in this row/col
                return self.set_hdr(ax, e.loc(ax), Some(e.index));
            }
        };
        let off_ax = ax.other();
        if self[head_idx].loc(off_ax) > e.loc(off_ax) {
            // `e` is the new first element
            e.set_next(ax, head_ptr);
            return self.set_hdr(ax, e.loc(ax), Some(e.index));
        }

        // `e` comes after at least one Element.  Search for its position.
        let mut prev = head_idx;
        while let Some(next) = self[prev].next(ax) {
            if self[next].loc(off_ax) >= e.loc(off_ax) {
                break;
            }
            prev = next;
        }
        // And splice it in-between `prev` and `nxt`
        e.set_next(ax, self[prev].next(ax));
        self[prev].set_next(ax, Some(e.index));
    }
    fn add_fillin(&mut self, row: usize, col: usize) -> Eindex {
        let zero = Complex64 { re: 0.0, im: 0.0 };
        return self._add_element(row, col, zero, true);
    }
    fn _add_element(&mut self, row: usize, col: usize, val: Complex64, fillin: bool) -> Eindex {
        // Element creation & insertion, used by `add_fillin` and the public `add_element`.
        let index = Eindex(self.elements.len());
        let mut e = Element::new(index.clone(), row, col, val, fillin);
        self.insert(&mut e);
        self.elements.push(e);
        return index;
    }
    /// Returns the Element-index at `(row, col)` if present, or None if not.
    pub fn get_elem(&self, row: usize, col: usize) -> Option<Eindex> {
        if row >= self.num_rows() {
            return None;
        }
        if col >= self.num_cols() {
            return None;
        }

        if row == col {
            // On diagonal; easy access
            return self.diag[row];
        }
        // Off-diagonal. Search across `row`.
        let mut ep = self.hdr(ROWS, row);
        while let Some(ei) = ep {
            let e = &self[ei];
            if e.col == col {
                return Some(ei);
            } else if e.col > col {
                return None;
            }
            ep = e.next_in_row;
        }
        return None;
    }
    /// Returns the Element-value at `(row, col)` if present, or None if not.
    pub fn get(&self, row: usize, col: usize) -> Option<Complex64> {
        return match self.get_elem(row, col) {
            None => None,
            Some(ei) => Some(self[ei].val),
        };
    }
    /// Make major state transitions
    fn set_state(&mut self, state: MatrixState) -> Result<(), &'static str> {
        match state {
            MatrixState::CREATED => return Ok(()), //Err("Matrix State Error"),
            MatrixState::FACTORING => {
                if self.state == MatrixState::FACTORING {
                    return Ok(());
                }
                if self.state == MatrixState::FACTORED {
                    return Err("Already Factored");
                }

                self.axes[Axis::ROWS].setup_factoring();
                self.axes[Axis::COLS].setup_factoring();

                self.state = state;
                return Ok(());
            }
            MatrixState::FACTORED => {
                if self.state == MatrixState::FACTORING {
                    self.state = state;
                    return Ok(());
                } else {
                    return Err("Matrix State Error");
                }
            }
        }
    }
    fn move_element(&mut self, ax: Axis, idx: Eindex, to: usize) {
        let loc = self[idx].loc(ax);
        if loc == to {
            return;
        }
        let off_ax = ax.other();
        let y = self[idx].loc(off_ax);

        if loc < to {
            let br = match self.before_loc(off_ax, y, to, Some(idx)) {
                Some(ei) => ei,
                None => panic!("ERROR"),
            };
            if br != idx {
                let be = self.prev(off_ax, idx, None);
                let nxt = self[idx].next(off_ax);
                match be {
                    None => self.set_hdr(off_ax, y, nxt),
                    Some(be) => self[be].set_next(off_ax, nxt),
                };
                let brn = self[br].next(off_ax);
                self[idx].set_next(off_ax, brn);
                self[br].set_next(off_ax, Some(idx));
            }
        } else {
            let br = self.before_loc(off_ax, y, to, None);
            let be = self.prev(off_ax, idx, None);

            if br != be {
                // We (may) need some pointer updates
                if let Some(ei) = be {
                    let nxt = self[idx].next(off_ax);
                    self[ei].set_next(off_ax, nxt);
                }
                match br {
                    None => {
                        // New first in row/col
                        let first = self.hdr(off_ax, y);
                        self[idx].set_next(off_ax, first);
                        self.axes[off_ax].hdrs[y] = Some(idx);
                    }
                    Some(br) => {
                        if br != idx {
                            // Splice `idx` in after `br`
                            let nxt = self[br].next(off_ax);
                            self[idx].set_next(off_ax, nxt);
                            self[br].set_next(off_ax, Some(idx));
                        }
                    }
                };
            }
        }

        // Update the moved-Element's location
        self[idx].set_loc(ax, to);

        if loc == y {
            // If idx was on our diagonal, remove it
            self.diag[loc] = None;
        } else if to == y {
            // Or if it's now on the diagonal, add it
            self.diag[to] = Some(idx);
        }
    }
    fn exchange_elements(&mut self, ax: Axis, ix: Eindex, iy: Eindex) {
        // Swap two elements `ax` indices.
        // Elements must be in the same off-axis vector,
        // and the first argument `ex` must be the lower-indexed off-axis.
        // E.g. exchange_elements(Axis.rows, ex, ey) exchanges the rows of ex and ey.

        let off_ax = ax.other();
        let off_loc = self[ix].loc(off_ax);

        let bx = self.prev(off_ax, ix, None);
        let by = match self.prev(off_ax, iy, Some(ix)) {
            Some(e) => e,
            None => panic!("ERROR!"),
        };

        let locx = self[ix].loc(ax);
        let locy = self[iy].loc(ax);
        self[iy].set_loc(ax, locx);
        self[ix].set_loc(ax, locy);

        match bx {
            None => {
                // If `ex` is the *first* entry in the column, replace it to our header-list
                self.set_hdr(off_ax, off_loc, Some(iy));
            }
            Some(bxe) => {
                // Otherwise patch ey into bx
                self[bxe].set_next(off_ax, Some(iy));
            }
        }

        if by == ix {
            // `ex` and `ey` are adjacent
            let tmp = self[iy].next(off_ax);
            self[iy].set_next(off_ax, Some(ix));
            self[ix].set_next(off_ax, tmp);
        } else {
            // Elements in-between `ex` and `ey`.  Update the last one.
            let xnxt = self[ix].next(off_ax);
            let ynxt = self[iy].next(off_ax);
            self[iy].set_next(off_ax, xnxt);
            self[ix].set_next(off_ax, ynxt);
            self[by].set_next(off_ax, Some(ix));
        }

        // Update our diagonal array, if necessary
        if locx == off_loc {
            self.diag[off_loc] = Some(iy);
        } else if locy == off_loc {
            self.diag[off_loc] = Some(ix);
        }
    }
    fn prev(&self, ax: Axis, idx: Eindex, hint: Option<Eindex>) -> Option<Eindex> {
        // Find the element previous to `idx` along axis `ax`.
        // If provided, `hint` *must* be before `idx`, or search will fail.
        let prev: Option<Eindex> = match hint {
            Some(_) => hint,
            None => self.hdr(ax, self[idx].loc(ax)),
        };
        let mut pi: Eindex = match prev {
            None => {
                return None;
            }
            Some(pi) if pi == idx => {
                return None;
            }
            Some(pi) => pi,
        };
        while let Some(nxt) = self[pi].next(ax) {
            if nxt == idx {
                break;
            }
            pi = nxt;
        }
        return Some(pi);
    }
    fn before_loc(
        &self,
        ax: Axis,
        loc: usize,
        before: usize,
        hint: Option<Eindex>,
    ) -> Option<Eindex> {
        let prev: Option<Eindex> = match hint {
            Some(_) => hint,
            None => self.hdr(ax, loc),
        };
        let off_ax = ax.other();
        let mut pi: Eindex = match prev {
            None => {
                return None;
            }
            Some(pi) if self[pi].loc(off_ax) >= before => {
                return None;
            }
            Some(pi) => pi,
        };
        while let Some(nxt) = self[pi].next(ax) {
            if self[nxt].loc(off_ax) >= before {
                break;
            }
            pi = nxt;
        }
        return Some(pi);
    }
    fn swap(&mut self, ax: Axis, a: usize, b: usize) {
        if a == b {
            return;
        }
        let x = min(a, b);
        let y = max(a, b);

        let hdrs = &self.axes[ax].hdrs;
        let mut ix = hdrs[x];
        let mut iy = hdrs[y];
        let off_ax = ax.other();

        loop {
            match (ix, iy) {
                (Some(ex), Some(ey)) => {
                    let ox = self[ex].loc(off_ax);
                    let oy = self[ey].loc(off_ax);
                    if ox < oy {
                        self.move_element(ax, ex, y);
                        ix = self[ex].next(ax);
                    } else if oy < ox {
                        self.move_element(ax, ey, x);
                        iy = self[ey].next(ax);
                    } else {
                        self.exchange_elements(ax, ex, ey);
                        ix = self[ex].next(ax);
                        iy = self[ey].next(ax);
                    }
                }
                (None, Some(ey)) => {
                    self.move_element(ax, ey, x);
                    iy = self[ey].next(ax);
                }
                (Some(ex), None) => {
                    self.move_element(ax, ex, y);
                    ix = self[ex].next(ax);
                }
                (None, None) => {
                    break;
                }
            }
        }
        // Swap all the relevant pointers & counters
        self.axes[ax].swap(x, y);
    }
    /// Updates self to S = L + U - I.
    /// Diagonal entries are those of U;
    /// L has diagonal entries equal to one.
    fn lu_factorize(&mut self) -> SpResult<()> {
        assert(self.diag.len()).gt(0);
        for k in 0..self.axes[ROWS].hdrs.len() {
            if self.hdr(ROWS, k).is_none() {
                return Err("Singular Matrix");
            }
        }
        for k in 0..self.axes[COLS].hdrs.len() {
            if self.hdr(COLS, k).is_none() {
                return Err("Singular Matrix");
            }
        }
        self.set_state(MatrixState::FACTORING)?;

        for n in 0..self.diag.len() - 1 {
            let pivot = match self.search_for_pivot(n) {
                None => return Err("Pivot Search Fail"),
                Some(p) => p,
            };
            self.swap(ROWS, self[pivot].row, n);
            self.swap(COLS, self[pivot].col, n);
            self.row_col_elim(pivot, n)?;
        }
        self.set_state(MatrixState::FACTORED)?;
        return Ok(());
    }

    fn search_for_pivot(&self, n: usize) -> Option<Eindex> {
        let mut ei = self.markowitz_search_diagonal(n);
        if let Some(_) = ei {
            return ei;
        }
        ei = self.markowitz_search_submatrix(n);
        if let Some(_) = ei {
            return ei;
        }
        return self.find_max(n);
    }

    fn max_after(&self, ax: Axis, after: Eindex) -> Eindex {
        let mut best = after;
        let mut best_val = self[after].val.norm();
        let mut e = self[after].next(ax);

        while let Some(ei) = e {
            let val = self[ei].val.norm();
            if val > best_val {
                best = ei;
                best_val = val;
            }
            e = self[ei].next(ax);
        }
        return best;
    }

    fn markowitz_product(&self, ei: Eindex) -> usize {
        let e = &self[ei];
        let mr = self[Axis::ROWS].markowitz[e.row];
        let mc = self[Axis::COLS].markowitz[e.col];
        assert(mr).gt(0);
        assert(mc).gt(0);
        return (mr - 1) * (mc - 1);
    }

    #[allow(non_snake_case)]
    fn markowitz_search_diagonal(&self, n: usize) -> Option<Eindex> {
        let REL_THRESHOLD = 1e-3;
        let ABS_THRESHOLD = 0.0;
        let TIES_MULT = 5;

        let mut best_elem = None;
        let mut best_mark = MAX; // Actually use usize::MAX!
        let mut best_ratio = 0.0;
        let mut num_ties = 0;

        for k in n..self.diag.len() {
            let d = match self.diag[k] {
                None => {
                    continue;
                }
                Some(d) => d,
            };

            // Check whether this element meets our threshold criteria
            let max_in_col = self.max_after(COLS, d);
            let threshold = REL_THRESHOLD * self[max_in_col].val.norm() + ABS_THRESHOLD;
            if self[d].val.norm() < threshold {
                continue;
            }

            // If so, compute and compare its Markowitz product to our best
            let mark = self.markowitz_product(d);
            if mark < best_mark {
                num_ties = 0;
                best_elem = self.diag[k];
                best_mark = mark;
                best_ratio = (self[d].val / self[max_in_col].val).norm();
            } else if mark == best_mark {
                num_ties += 1;
                let ratio = (self[d].val / self[max_in_col].val).norm();
                if ratio > best_ratio {
                    best_elem = self.diag[k];
                    best_mark = mark;
                    best_ratio = ratio;
                }
                if num_ties >= best_mark * TIES_MULT {
                    return best_elem;
                }
            }
        }
        return best_elem;
    }

    #[allow(non_snake_case)]
    fn markowitz_search_submatrix(&self, n: usize) -> Option<Eindex> {
        let REL_THRESHOLD = 1e-3;
        let ABS_THRESHOLD = 0.0;

        let mut best_elem = None;
        let mut best_mark = MAX; // Actually use usize::MAX!
        let mut best_ratio = 0.0;
        let mut _num_ties = 0;

        for _ in n..self.axes[COLS].hdrs.len() {
            let mut e = self.hdr(COLS, n);
            // Advance to a row ≥ n
            while let Some(ei) = e {
                if self[ei].row >= n {
                    break;
                }
                e = self[ei].next_in_col;
            }
            let ei = match e {
                None => {
                    continue;
                }
                Some(d) => d,
            };

            // Check whether this element meets our threshold criteria
            let max_in_col = self.max_after(COLS, ei);
            let _threshold = REL_THRESHOLD * self[max_in_col].val.norm() + ABS_THRESHOLD;

            while let Some(ei) = e {
                // If so, compute and compare its Markowitz product to our best
                let mark = self.markowitz_product(ei);
                if mark < best_mark {
                    _num_ties = 0;
                    best_elem = e;
                    best_mark = mark;
                    best_ratio = (self[ei].val / self[max_in_col].val).norm();
                } else if mark == best_mark {
                    _num_ties += 1;
                    let ratio = (self[ei].val / self[max_in_col].val).norm();
                    if ratio > best_ratio {
                        best_elem = e;
                        best_mark = mark;
                        best_ratio = ratio;
                    }
                    //                    // FIXME: do we want tie-counting in here?
                    //                    if _num_ties >= best_mark * TIES_MULT { return best_elem; }
                }
                e = self[ei].next_in_col;
            }
        }
        return best_elem;
    }
    /// Find the max (abs value) element in sub-matrix of indices ≥ `n`.
    /// Returns `None` if no elements present.
    fn find_max(&self, n: usize) -> Option<Eindex> {
        let mut max_elem = None;
        let mut max_val = 0.0;

        // Search each column ≥ n
        for k in n..self.axes[COLS].hdrs.len() {
            let mut ep = self.hdr(COLS, k);

            // Advance to a row ≥ n
            while let Some(ei) = ep {
                if self[ei].row >= n {
                    break;
                }
                ep = self[ei].next_in_col;
            }
            // And search over remaining elements
            while let Some(ei) = ep {
                let val = self[ei].val.norm();
                if val > max_val {
                    max_elem = ep;
                    max_val = val;
                }
                ep = self[ei].next_in_col;
            }
        }
        return max_elem;
    }

    fn row_col_elim(&mut self, pivot: Eindex, n: usize) -> SpResult<()> {
        let de = match self.diag[n] {
            Some(de) => de,
            None => return Err("Singular Matrix"),
        };
        assert(de).eq(pivot);
        let pivot_val = self[pivot].val;
        let zero = Complex64{re: 0.0, im: 0.0};
        assert(pivot_val).ne(zero);

        // Divide elements in the pivot column by the pivot-value
        let mut plower = self[pivot].next_in_col;
        while let Some(ple) = plower {
            self[ple].val /= pivot_val;
            plower = self[ple].next_in_col;
        }

        let mut pupper = self[pivot].next_in_row;
        while let Some(pue) = pupper {
            let pupper_col = self[pue].col;
            plower = self[pivot].next_in_col;
            let mut psub = self[pue].next_in_col;
            while let Some(ple) = plower {
                // Walk `psub` down to the lower pointer
                while let Some(pse) = psub {
                    if self[pse].row >= self[ple].row {
                        break;
                    }
                    psub = self[pse].next_in_col;
                }
                let pse = match psub {
                    None => self.add_fillin(self[ple].row, pupper_col),
                    Some(pse) if self[pse].row > self[ple].row => {
                        self.add_fillin(self[ple].row, pupper_col)
                    }
                    Some(pse) => pse,
                };

                // Update the `psub` element value
                let result = (self[pue].val) * (self[ple].val);
                self[pse].val -= result;
                psub = self[pse].next_in_col;
                plower = self[ple].next_in_col;
            }
            self.axes[COLS].markowitz[pupper_col] -= 1;
            pupper = self[pue].next_in_row;
        }
        // Update remaining Markowitz counts
        self.axes[ROWS].markowitz[n] -= 1;
        self.axes[COLS].markowitz[n] -= 1;
        plower = self[pivot].next_in_col;
        while let Some(ple) = plower {
            let plower_row = self[ple].row;
            self.axes[ROWS].markowitz[plower_row] -= 1;
            plower = self[ple].next_in_col;
        }
        return Ok(());
    }
    /// Solve the system `Ax=b`, where:
    /// * `A` is `self`
    /// * `b` is argument `rhs`
    /// * `x` is the return value.
    ///
    /// Returns a `Result` containing the `Vec<Complex64>` representing `x` if successful.
    /// Returns an `Err` if unsuccessful.
    ///
    /// Performs LU factorization, forward and backward substitution.
    pub fn solve(&mut self, rhs: Vec<Complex64>) -> SpResult<Vec<Complex64>> {
        if self.state == MatrixState::CREATED {
            self.lu_factorize()?;
        }
        assert(self.state).eq(MatrixState::FACTORED);

        // Unwind any row-swaps
        let zero = Complex64{re: 0.0, im: 0.0};
        let mut c: Vec<Complex64> = vec![zero; rhs.len()];
        let row_mapping = self.axes[ROWS].mapping.as_ref().unwrap();
        for k in 0..c.len() {
            c[row_mapping.e2i[k]] = rhs[k];
        }

        // Forward substitution: Lc=b
        for k in 0..self.diag.len() {
            // Walk down each column, update c
            if c[k] == zero {
                continue;
            } // No updates to make on this iteration

            // c[d.row] /= d.val

            let di = match self.diag[k] {
                Some(di) => di,
                None => return Err("Singular Matrix"),
            };
            let mut e = self[di].next_in_col;
            while let Some(ei) = e {
                let result = c[k] * self[ei].val;
                c[self[ei].row] -= result;
                e = self[ei].next_in_col;
            }
        }

        // Backward substitution: Ux=c
        for k in (0..self.diag.len()).rev() {
            // Walk each row, update c
            let di = match self.diag[k] {
                Some(di) => di,
                None => return Err("Singular Matrix"),
            };
            let mut ep = self[di].next_in_row;
            while let Some(ei) = ep {
                let result = c[self[ei].col] * self[ei].val;
                c[k] -= result;
                ep = self[ei].next_in_row;
            }
            c[k] /= self[di].val;
        }

        // Unwind any column-swaps
        let zero = Complex64{re: 0.0, im: 0.0};
        let mut soln: Vec<Complex64> = vec![zero; c.len()];
        let col_mapping = self.axes[COLS].mapping.as_ref().unwrap();
        for k in 0..c.len() {
            soln[k] = c[col_mapping.e2i[k]];
        }
        return Ok(soln);
    }
    fn hdr(&self, ax: Axis, loc: usize) -> Option<Eindex> {
        self.axes[ax].hdrs[loc]
    }
    fn set_hdr(&mut self, ax: Axis, loc: usize, ei: Option<Eindex>) {
        self.axes[ax].hdrs[loc] = ei;
    }
    fn _swap_rows(&mut self, x: usize, y: usize) {
        self.swap(ROWS, x, y)
    }
    fn _swap_cols(&mut self, x: usize, y: usize) {
        self.swap(COLS, x, y)
    }
    fn num_rows(&self) -> usize {
        self.axes[ROWS].hdrs.len()
    }
    fn num_cols(&self) -> usize {
        self.axes[COLS].hdrs.len()
    }
    fn _size(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }
}

impl Index<Eindex> for Matrix {
    type Output = Element;
    fn index(&self, index: Eindex) -> &Self::Output {
        &self.elements[index.0]
    }
}

impl IndexMut<Eindex> for Matrix {
    fn index_mut(&mut self, index: Eindex) -> &mut Self::Output {
        &mut self.elements[index.0]
    }
}

impl Index<Axis> for Matrix {
    type Output = AxisData;
    fn index(&self, ax: Axis) -> &Self::Output {
        &self.axes[ax]
    }
}

impl IndexMut<Axis> for Matrix {
    fn index_mut(&mut self, ax: Axis) -> &mut Self::Output {
        &mut self.axes[ax]
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "csparse21::Matrix (rows={}, cols={}, elems={}\n",
            self.num_rows(),
            self.num_cols(),
            self.elements.len()
        )?;
        for e in self.elements.iter() {
            write!(f, "({}, {}, {}) \n", e.row, e.col, e.val)?;
        }
        write!(f, "\n")
    }
}

#[derive(Debug, Clone)]
struct NonRealNumError;

impl fmt::Display for NonRealNumError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid first item to double")
    }
}

impl Error for NonRealNumError {
    fn description(&self) -> &str {
        "invalid first item to double"
    }

    fn cause(&self) -> Option<&dyn Error> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

/// Sparse Matrix System
///
/// Represents a linear system of the form `Ax=b`
///
#[allow(dead_code)]
pub struct System {
    mat: Matrix,
    rhs: Vec<Complex64>,
    title: Option<String>,
    size: usize,
}

impl System {
    /// Splits a `System` into a two-tuple of `self.matrix` and `self.rhs`.
    /// Nothing is copied; `self` is consumed in the process.
    pub fn split(self) -> (Matrix, Vec<Complex64>) {
        (self.mat, self.rhs)
    }

    /// Solve the system `Ax=b`, where:
    /// * `A` is `self.matrix`
    /// * `b` is `self.rhs`
    /// * `x` is the return value.
    ///
    /// Returns a `Result` containing the `Vec<Complex64>` representing `x` if successful.
    /// Returns an `Err` if unsuccessful.
    ///
    /// Performs LU factorization, forward and backward substitution.
    pub fn solve(mut self) -> SpResult<Vec<Complex64>> {
        self.mat.solve(self.rhs)
    }
}

struct Assert<T> {
    val: T,
}

fn assert<T>(val: T) -> Assert<T> {
    Assert { val }
}

impl<T> Assert<T> {
    fn raise(&self) {
        // Breakpoint here
        panic!("Assertion Failed");
    }
}

impl<T: PartialEq> Assert<T> {
    fn eq(&self, other: T) {
        if self.val != other {
            self.raise();
        }
    }
    fn ne(&self, other: T) {
        if self.val == other {
            self.raise();
        }
    }
}

#[allow(dead_code)]
impl<T: PartialOrd> Assert<T> {
    fn gt(&self, other: T) {
        if self.val <= other {
            self.raise();
        }
    }
    fn lt(&self, other: T) {
        if self.val >= other {
            self.raise();
        }
    }
    fn ge(&self, other: T) {
        if self.val < other {
            self.raise();
        }
    }
    fn le(&self, other: T) {
        if self.val > other {
            self.raise();
        }
    }
}
