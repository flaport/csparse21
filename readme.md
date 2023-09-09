# csparse21

Solving large systems of **complex-valued**\* linear equations using sparse matrix methods.

\* this is a fork\*\* of [sparse21](https://github.com/dan-fritchman/sparse21) for
complex-valued sparse matrices.

\*\* it would probably be better to make sparse21 work for generic type arguments but I don't have time for that.

```rust
use num_complex::Complex64;
let mut m = csparse21::Matrix::from_entries(vec![
            (0, 0, Complex64{re: 1.0 , im: 1.0}),
            (0, 1, Complex64{re: 1.0 , im: 1.0}),
            (0, 2, Complex64{re: 1.0 , im: 1.0}),
            (1, 1, Complex64{re: 2.0 , im: 1.0}),
            (1, 2, Complex64{re: 5.0 , im: 1.0}),
            (2, 0, Complex64{re: 2.0 , im: 1.0}),
            (2, 1, Complex64{re: 5.0 , im: 1.0}),
            (2, 2, Complex64{re: -1.0, im: 1.0}),
        ]);

        let soln = m.solve(vec![
          Complex64{re: 6.0, im: 5.0},
          Complex64{re:-4.0, im: 27.0},
          Complex64{re: 5.0, im: -5.0},
        ]);
```

Sparse methods are primarily valuable for systems in which the number of non-zero entries is substantially less than the overall size of the matrix. Such situations are common in physical systems, including electronic circuit simulation. All elements of a sparse matrix are assumed to be zero-valued unless indicated otherwise.

## Usage

CSparse21 exposes two primary data structures:

* `Matrix` represents an `Complex64`-valued sparse matrix
* `System` represents a system of linear equations of the form `Ax=b`, including a `Matrix` (A) and right-hand-side `Vec` (b).

Once matrices and systems have been created, their primary public method is `solve`, which returns a (dense) `Vec` solution-vector.


## Matrix

CSparse21 matrices can be constructed from a handful of data-sources

`Matrix::new` creates an empty matrix, to which elements can be added via the `add_element` and `add_elements` methods.

```rust
let mut m = Matrix::new();

m.add_element(0, 0, Complex64{re:11.0, im: 12.0});
m.add_element(7, 0, Complex64{re:22.0, im: 23.0});
m.add_element(0, 7, Complex64{re:33.0, im: 0.0});
m.add_element(7, 7, Complex64{re:44.0, im: -3.0});
```

```rust
let mut m = Matrix::new();

m.add_elements(vec![
    (0, 0, Complex64{re: 1.0 , im: 1.0}),
    (0, 1, Complex64{re: 1.0 , im: 1.0}),
    (0, 2, Complex64{re: 1.0 , im: 1.0}),
    (2, 1, Complex64{re: 5.0 , im: 1.0}),
    (2, 2, Complex64{re: -1.0, im: 1.0}),
]);
```

The arguments to `add_element` are a row (`usize`), column (`usize`), and value (`Complex64`).
Adding elements (plural) via `add_elements` takes a vector of `(usize, usize, Complex64)` tuples, representing the row, col, and val.

Unlike common mathematical notation, all locations in `csparse21` matrices and vectors are zero-indexed.
Adding a non-zero at the "first" matrix element therefore implies calling `add_element(0, 0, val)`.

Creating a `Matrix` from data entries with `Matrix::from_entries`:

```rust
let mut m = Matrix::from_entries(vec![
    (0, 0, Complex64{re: 1.0 , im: 1.0}),
    (0, 1, Complex64{re: 1.0 , im: 1.0}),
    (0, 2, Complex64{re: 1.0 , im: 1.0}),
    (1, 1, Complex64{re: 2.0 , im: 1.0}),
    (1, 2, Complex64{re: 5.0 , im: 1.0}),
    (2, 0, Complex64{re: 2.0 , im: 1.0}),
    (2, 1, Complex64{re: 5.0 , im: 1.0}),
    (2, 2, Complex64{re: -1.0, im: 1.0}),
]);
```

The `Matrix::identity` method returns a new identity matrix of size (n x n):

```rust
let mut m = Matrix::identity(3);
```

### Solving

CSparse21 matrices are built for solving equation-systems. The primary public method of a `Matrix` is `solve()`, which accepts a `Vec` right-hand-side as its sole argument, and returns a solution `Vec` of the same size.

### Matrix Mutability

You may have noticed all examples to date declare matrices as `mut`, perhaps unnecessarily. This is on purpose. The `Matrix::solve` method (un-rustily) modifies the matrix *in-place*. For larger matrices, the in-place modification saves orders of magnitude of memory, as well as time creating and destroying elements. While in-place self-modification falls out of line with the Rust ethos, it follows a long lineage of scientific computing tools for this and similar tasks.

So: in order to be solved, matrices must be declared `mut`.

