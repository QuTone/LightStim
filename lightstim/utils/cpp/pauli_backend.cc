// Unsigned inverse-tableau algebra for the existing record-carrying Tracker.
//
// The inverse query and generator re-basing below are adapted from Hou Yuewen
// (isaachyw)'s record-tableau engine, QuTone/LightStim PR #110, commit 37df61c.
// Original author: isaachyw <isaachyw@umich.edu>. Apache-2.0 (repository LICENSE).
// Tracker retains detector/logical policy and prescribes source rows/pivots.
// Native kernels execute its exact role algebra and record transformations.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <bit>
#include <memory>
#include <stdexcept>
#include <vector>

#include "stim/mem/simd_word.h"
#include "stim/stabilizers/tableau.h"
#include "stim/stabilizers/tableau_transposed_raii.h"
#include "pauli_propagation.h"

namespace py = pybind11;
using Rows = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using Indices = std::vector<size_t>;
constexpr size_t W = stim::MAX_BITWORD_WIDTH;

class CommutingBasis {
    size_t n, count;
    stim::Tableau<W> inv;
    // Known generator -> earliest independent ORIGINAL source row. Completing
    // the tableau supplies other generators, which must never count as known.
    std::vector<int64_t> source;
    bool valid = true;

    static size_t width(const Rows &rows) {
        if (rows.ndim() != 2 || rows.shape(1) % 2)
            throw std::invalid_argument("Pauli basis must be a 2D even-width matrix");
        return (size_t)rows.shape(1) / 2;
    }

    // Valid only inside TableauTransposedRaii. C^-1 P C = X^anti Z^decomp.
    // XOR of single-qubit queries; phases intentionally do not enter GF(2).
    void query(const uint8_t *row, Indices &anti, Indices &decomp) const {
        anti.clear();
        decomp.clear();
        // Iterate physical support once, then gather inverse coordinates.
        Indices x, z;
        for (size_t q = 0; q < n; q++) {
            if (row[q] & 1) x.push_back(q);
            if (row[n + q] & 1) z.push_back(q);
        }
        for (size_t k = 0; k < n; k++) {
            bool a = false, b = false;
            for (size_t q : x) {
                a ^= inv.xs.xt[k][q];
                b ^= inv.xs.zt[k][q];
            }
            for (size_t q : z) {
                a ^= inv.zs.xt[k][q];
                b ^= inv.zs.zt[k][q];
            }
            if (a) anti.push_back(k);
            if (b) decomp.push_back(k);
        }
    }

    // Re-base one previously unknown generator to P. Adapted from PR #110's
    // Engine::make_generator. Known generators remain unchanged.
    void anchor(stim::TableauTransposedRaii<W> &trans, size_t g,
                const Indices &anti, const Indices &decomp) {
        if (!anti.empty()) {
            if (anti.size() != 1 || anti[0] != g)
                throw std::logic_error("Inverse-tableau anchor has an invalid pivot");
            bool y = false;
            for (size_t k : decomp) {
                if (k == g) y = true;
                else trans.append_ZCZ(g, k);
            }
            if (y) trans.append_H_YZ(g);
            else trans.append_H_XZ(g);
        } else {
            for (size_t k : decomp)
                if (k != g) trans.append_ZCX(k, g);
        }
    }

    // Append source rows without rebasing generators already known. Original
    // source positions include redundant rows, so rank is not a row offset.
    void append(const Rows &rows) {
        if (width(rows) != n)
            throw std::invalid_argument("Extended rows must have the basis register width");
        const size_t old_count = count, added_count = rows.shape(0);
        count += added_count;
        if (!valid || !added_count) return;
        stim::TableauTransposedRaii<W> trans(inv);
        Indices anti, decomp;
        for (size_t r = 0; r < added_count; r++) {
            const uint8_t *row = rows.data() + r * 2 * n;
            query(row, anti, decomp);
            for (size_t k : anti) {
                if (source[k] >= 0) {
                    valid = false;  // Noncommuting space: reference GF(2) path.
                    return;
                }
            }
            size_t g = n;
            if (!anti.empty()) {
                g = anti.front();
                for (size_t k : anti)
                    if (k != g) trans.append_ZCX(g, k);
                query(row, anti, decomp);
            } else {
                for (size_t k : decomp)
                    if (source[k] < 0) { g = k; break; }
            }
            if (g == n) continue;  // Redundant unsigned row (including I).
            anchor(trans, g, anti, decomp);
            source[g] = (int64_t)(old_count + r);
        }
    }

public:
    explicit CommutingBasis(const Rows &basis)
        : n(width(basis)), count(0), inv(n), source(n, -1) {
        append(basis);
    }

    CommutingBasis extended(const Rows &rows) const {
        // Stim's tableau and its packed bit arrays have deep-copy semantics.
        // A cached parent stays immutable, including when the suffix fails
        // the commuting check and the resulting child needs the fallback.
        CommutingBasis result(*this);
        result.append(rows);
        return result;
    }

    bool commuting() const { return valid; }
    size_t estimated_bytes() const {
        // Conservative padded tableau size plus source map (excludes caller key).
        size_t padded = ((n + W - 1) / W) * W;
        return padded * padded / 2 + 4 * padded + n * sizeof(int64_t);
    }

    py::tuple decompose(const Rows &targets) {
        if (targets.ndim() != 2 || (size_t)targets.shape(1) != 2 * n)
            throw std::invalid_argument("Targets must have the basis register width");
        if (!valid) throw std::invalid_argument("Basis is not commuting");
        const size_t m = targets.shape(0), words = (2 * n + 63) / 64;
        py::array_t<uint8_t> coeff({(py::ssize_t)m, (py::ssize_t)count});
        py::array_t<bool> dependent(m);
        std::fill_n(coeff.mutable_data(), m * count, uint8_t{0});
        std::fill_n(dependent.mutable_data(), m, true);
        std::vector<size_t> new_indices;
        // Only new target dimensions need elimination. Their coordinates are
        // modulo known Z generators, so no physical-state RREF is constructed.
        std::vector<std::vector<uint64_t>> pivots(2 * n);
        stim::TableauTransposedRaii<W> trans(inv);
        Indices anti, decomp;
        for (size_t r = 0; r < m; r++) {
            query(targets.data() + r * 2 * n, anti, decomp);
            bool known = anti.empty();
            for (size_t k : decomp) known &= source[k] >= 0;
            dependent.mutable_data()[r] = known;
            if (known) {
                for (size_t k : decomp)
                    coeff.mutable_data()[r * count + source[k]] = 1;
                continue;
            }
            std::vector<uint64_t> residual(words, 0);
            for (size_t k : anti) residual[k / 64] ^= uint64_t{1} << (k % 64);
            for (size_t k : decomp)
                if (source[k] < 0)
                    residual[(n + k) / 64] ^= uint64_t{1} << ((n + k) % 64);
            for (size_t p = 0; p < 2 * n; p++) {
                if (!(residual[p / 64] & (uint64_t{1} << (p % 64)))) continue;
                if (pivots[p].empty()) {
                    pivots[p] = std::move(residual);
                    new_indices.push_back(r);
                    break;
                }
                for (size_t w = 0; w < words; w++) residual[w] ^= pivots[p][w];
            }
        }
        return py::make_tuple(coeff, dependent, new_indices);
    }
};

#include "packed_row_algebra.h"
#include "persistent_record_state.h"

PYBIND11_MODULE(_pauli_backend_cpp, mod) {
    mod.doc() = "Unsigned Pauli algebra, inverse-tableau core adapted from Hou Yuewen's PR #110";
    py::class_<CommutingBasis>(mod, "CommutingBasis")
        .def(py::init<const Rows &>())
        .def_property_readonly("commuting", &CommutingBasis::commuting)
        .def_property_readonly("estimated_bytes", &CommutingBasis::estimated_bytes)
        .def("extended", &CommutingBasis::extended)
        .def("decompose", &CommutingBasis::decompose);
    mod.def("propagate_paulis", &propagate_paulis);
    lightstim_packed::bind(mod);
    bind_persistent_record_state(mod);
}
