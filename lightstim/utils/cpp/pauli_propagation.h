#pragma once

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Forward Pauli propagation is a GF(2) matrix product. Packing the right-hand
// rows lets each nonzero coefficient XOR 64 output columns at a time instead
// of using NumPy's dense uint8 arithmetic. This changes only the representation
// used for multiplication; row order and all tracker metadata remain intact.
inline pybind11::array_t<uint8_t> propagate_paulis(
    pybind11::array_t<uint8_t,
        pybind11::array::c_style | pybind11::array::forcecast> rows,
    pybind11::array_t<uint8_t,
        pybind11::array::c_style | pybind11::array::forcecast> matrix) {
    if (rows.ndim() != 2 || matrix.ndim() != 2) {
        throw std::invalid_argument("Pauli propagation requires two 2D matrices.");
    }
    if (rows.shape(1) != matrix.shape(0)) {
        throw std::invalid_argument(
            "Pauli propagation matrices have incompatible dimensions.");
    }

    const auto num_rows = rows.shape(0);
    const auto inner_size = rows.shape(1);
    const auto num_columns = matrix.shape(1);
    pybind11::array_t<uint8_t> output({num_rows, num_columns});
    if (num_rows == 0 || num_columns == 0) {
        return output;
    }

    const auto num_words = (num_columns + 63) / 64;
    const uint8_t *row_data = rows.data();
    const uint8_t *matrix_data = matrix.data();
    uint8_t *output_data = output.mutable_data();
    {
        pybind11::gil_scoped_release release;
        std::vector<uint64_t> packed_matrix(inner_size * num_words, 0);
        for (pybind11::ssize_t row = 0; row < inner_size; ++row) {
            for (pybind11::ssize_t column = 0; column < num_columns; ++column) {
                if (matrix_data[row * num_columns + column] & 1) {
                    packed_matrix[row * num_words + column / 64] |=
                        uint64_t{1} << (column % 64);
                }
            }
        }

        std::vector<uint64_t> accumulated(num_words);
        for (pybind11::ssize_t row = 0; row < num_rows; ++row) {
            std::fill(accumulated.begin(), accumulated.end(), uint64_t{0});
            for (pybind11::ssize_t k = 0; k < inner_size; ++k) {
                if (!(row_data[row * inner_size + k] & 1)) {
                    continue;
                }
                const auto *source = packed_matrix.data() + k * num_words;
                for (pybind11::ssize_t word = 0; word < num_words; ++word) {
                    accumulated[word] ^= source[word];
                }
            }
            for (pybind11::ssize_t column = 0; column < num_columns; ++column) {
                output_data[row * num_columns + column] =
                    (accumulated[column / 64] >> (column % 64)) & 1;
            }
        }
    }
    return output;
}
