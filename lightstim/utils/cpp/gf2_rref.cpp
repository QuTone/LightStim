/**
 * GF(2) Gaussian Elimination — bitpacked for high performance.
 *
 * Operates on binary matrices packed into uint64_t words (64 columns per word).
 * Row XOR = single bitwise ^ per word → 64x fewer ops than element-wise.
 *
 * Exposed to Python via pybind11 as _gf2_rref_cpp module.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

/**
 * Pack a (m, n) uint8 bool matrix into (m, n_words) uint64_t words.
 * Each word holds 64 consecutive columns, little-endian bit order within word
 * (column j → bit j%64 of word j/64).
 */
static std::vector<uint64_t> pack_matrix(const uint8_t* data, int m, int n, int& n_words) {
    n_words = (n + 63) / 64;
    std::vector<uint64_t> packed(m * n_words, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (data[i * n + j]) {
                packed[i * n_words + j / 64] |= (1ULL << (j % 64));
            }
        }
    }
    return packed;
}

/**
 * Unpack (m, n_words) uint64_t words back to (m, n) uint8 matrix.
 */
static void unpack_matrix(const std::vector<uint64_t>& packed, uint8_t* out, int m, int n, int n_words) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = (packed[i * n_words + j / 64] >> (j % 64)) & 1;
        }
    }
}

/**
 * In-place GF(2) (reduced) row echelon form on a bitpacked matrix.
 *
 * Args:
 *   packed: row-major (m, n_words) bitpacked matrix, modified in-place.
 *   transform: row-major (m, m_words) identity matrix for tracking row ops,
 *              modified in-place. m_words = (m+63)/64.
 *   m, n: logical dimensions.
 *   n_words: number of uint64 words per row.
 *   reduced: if true, do full reduced row echelon (clear above and below pivot).
 *
 * Returns:
 *   (rank, pivot_cols)
 */
static std::pair<int, std::vector<int>> rref_packed(
    std::vector<uint64_t>& packed,
    std::vector<uint64_t>& transform,
    int m, int n, int n_words, int m_words, bool reduced
) {
    int pivot_row = 0;
    std::vector<int> pivot_cols;

    for (int col = 0; col < n && pivot_row < m; col++) {
        int word_idx = col / 64;
        uint64_t bit_mask = 1ULL << (col % 64);

        // Find pivot in this column (at or below pivot_row)
        int swap_row = -1;
        for (int i = pivot_row; i < m; i++) {
            if (packed[i * n_words + word_idx] & bit_mask) {
                swap_row = i;
                break;
            }
        }
        if (swap_row < 0) continue; // all-zero column below pivot

        // Swap rows if needed
        if (swap_row != pivot_row) {
            for (int w = 0; w < n_words; w++)
                std::swap(packed[swap_row * n_words + w], packed[pivot_row * n_words + w]);
            for (int w = 0; w < m_words; w++)
                std::swap(transform[swap_row * m_words + w], transform[pivot_row * m_words + w]);
        }

        // Eliminate other rows
        int start = reduced ? 0 : pivot_row + 1;
        for (int i = start; i < m; i++) {
            if (i == pivot_row) continue;
            if (packed[i * n_words + word_idx] & bit_mask) {
                // Row XOR: row[i] ^= row[pivot_row]
                for (int w = 0; w < n_words; w++)
                    packed[i * n_words + w] ^= packed[pivot_row * n_words + w];
                for (int w = 0; w < m_words; w++)
                    transform[i * m_words + w] ^= transform[pivot_row * m_words + w];
            }
        }

        pivot_cols.push_back(col);
        pivot_row++;
    }

    return {pivot_row, pivot_cols};
}


/**
 * Python-facing function: row_echelon over GF(2).
 *
 * Input:  (m, n) numpy uint8 array (0/1 values).
 * Output: tuple(row_ech_form, rank, transform, pivot_cols)
 *         matching the Python row_echelon() signature exactly.
 */
static py::tuple row_echelon_cpp(py::array_t<uint8_t> mat, bool reduced) {
    auto buf = mat.request();
    if (buf.ndim != 2)
        throw std::runtime_error("row_echelon_cpp: input must be 2D");

    int m = buf.shape[0];
    int n = buf.shape[1];
    const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);

    // Pack matrix and identity transform
    int n_words;
    auto packed = pack_matrix(data, m, n, n_words);

    int m_words = (m + 63) / 64;
    std::vector<uint64_t> transform(m * m_words, 0);
    for (int i = 0; i < m; i++) {
        transform[i * m_words + i / 64] |= (1ULL << (i % 64));
    }

    // RREF
    auto [rank, pivot_cols] = rref_packed(packed, transform, m, n, n_words, m_words, reduced);

    // Unpack results
    auto result_mat = py::array_t<uint8_t>({m, n});
    unpack_matrix(packed, static_cast<uint8_t*>(result_mat.request().ptr), m, n, n_words);

    auto result_transform = py::array_t<uint8_t>({m, m});
    unpack_matrix(transform, static_cast<uint8_t*>(result_transform.request().ptr), m, m, m_words);

    // Convert to int arrays (matching Python signature)
    auto result_mat_int = py::array_t<int>(result_mat.size());
    auto result_transform_int = py::array_t<int>(result_transform.size());
    result_mat_int.resize({m, n});
    result_transform_int.resize({m, m});
    {
        auto rm = result_mat.unchecked<2>();
        auto ri = result_mat_int.mutable_unchecked<2>();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                ri(i, j) = rm(i, j);
    }
    {
        auto rm = result_transform.unchecked<2>();
        auto ri = result_transform_int.mutable_unchecked<2>();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                ri(i, j) = rm(i, j);
    }

    py::list pivot_list;
    for (int p : pivot_cols) pivot_list.append(p);

    return py::make_tuple(result_mat_int, rank, result_transform_int, pivot_list);
}


PYBIND11_MODULE(_gf2_rref_cpp, mod) {
    mod.doc() = "High-performance GF(2) Gaussian elimination via bitpacked uint64 rows.";
    mod.def("row_echelon", &row_echelon_cpp,
            py::arg("mat"), py::arg("reduced") = false,
            "GF(2) (reduced) row echelon form. Same signature as linear_algebra.row_echelon.");
}
