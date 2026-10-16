// Packed coordinate algebra for the S_bare/S_dressed/L tracker.
// Complements Hou Yuewen's inverse-tableau architecture (LightStim PR #110)
// with source-lineage-aware operations on arbitrary, possibly noncommuting,
// role spaces. No commutativity assumption is made by this module.
#pragma once

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <utility>

namespace lightstim_packed {
using Bits = std::vector<uint64_t>;
inline size_t words(size_t n) { return (n + 63) / 64; }
inline bool bit(const Bits &v, size_t k) { return (v[k >> 6] >> (k & 63)) & 1; }
inline void set_bit(Bits &v, size_t k) { v[k >> 6] |= uint64_t{1} << (k & 63); }
inline void xor_into(Bits &a, const Bits &b) {
    for (size_t k = 0; k < a.size(); k++) a[k] ^= b[k];
}
inline bool nonzero(const Bits &v) {
    return std::any_of(v.begin(), v.end(), [](uint64_t w) { return w != 0; });
}
inline size_t first_bit(const Bits &v, size_t limit) {
    for (size_t k = 0; k < v.size(); k++)
        if (v[k]) return k * 64 + std::countr_zero(v[k]);
    return limit;
}
inline Bits unit(size_t n, size_t k) { Bits out(words(n)); set_bit(out, k); return out; }

struct PackedRows {
    size_t width;
    std::vector<Bits> data;
    explicit PackedRows(size_t width = 0) : width(width) {}
    explicit PackedRows(const Rows &rows) : width(rows.ndim() == 2 ? rows.shape(1) : 0) {
        if (rows.ndim() != 2) throw std::invalid_argument("Rows must be a two-dimensional matrix");
        data.resize(rows.shape(0), Bits(words(width)));
        for (size_t r = 0; r < data.size(); r++)
            for (size_t c = 0; c < width; c++)
                if (rows.data()[r * width + c] & 1) set_bit(data[r], c);
    }
    size_t size() const { return data.size(); }
    py::array_t<uint8_t> numpy() const {
        py::array_t<uint8_t> result({(py::ssize_t)size(), (py::ssize_t)width});
        std::fill_n(result.mutable_data(), size() * width, uint8_t{0});
        for (size_t r = 0; r < size(); r++)
            for (size_t w = 0; w < data[r].size(); w++) {
                auto v = data[r][w];
                while (v) {
                    size_t c = w * 64 + std::countr_zero(v);
                    if (c < width) result.mutable_data()[r * width + c] = 1;
                    v &= v - 1;
                }
            }
        return result;
    }
    PackedRows select(const Indices &indices) const {
        PackedRows result(width);
        for (size_t i : indices) {
            if (i >= size()) throw std::out_of_range("Packed row index out of range");
            result.data.push_back(data[i]);
        }
        return result;
    }
    void append(const PackedRows &other) {
        if (other.width != width) throw std::invalid_argument("Row widths differ");
        data.insert(data.end(), other.data.begin(), other.data.end());
    }
};

// Echelon rows carry original-source coefficients. Insertion in source order
// makes every dependency use the earliest independent original rows, exactly
// the free-variable convention of RREF(matrix.T), without forming that RREF.
class Basis {
    size_t width_, source_count_;
    std::vector<Bits> pivots_, lineage_;
    Indices selected_;
public:
    Basis(size_t width, size_t source_count)
        : width_(width), source_count_(source_count), pivots_(width), lineage_(width) {}
    explicit Basis(const PackedRows &rows) : Basis(rows.width, rows.size()) {
        for (size_t r = 0; r < rows.size(); r++) insert(rows.data[r], unit(rows.size(), r), r);
    }
    Basis extended(const PackedRows &rows) const {
        if (rows.width != width_) throw std::invalid_argument("Row widths differ");
        Basis result(*this);
        size_t previous_count = result.source_count_;
        result.source_count_ += rows.size();
        for (size_t p = 0; p < result.width_; p++)
            if (!result.pivots_[p].empty()) result.lineage_[p].resize(words(result.source_count_));
        for (size_t i = 0; i < rows.size(); i++)
            result.insert(rows.data[i], unit(result.source_count_, previous_count + i), previous_count + i);
        return result;
    }
    size_t rank() const { return selected_.size(); }
    size_t width() const { return width_; }
    size_t source_count() const { return source_count_; }
    const Indices &selected() const { return selected_; }
    void reduce(Bits &row, Bits *coeff = nullptr) const {
        // Source coordinates are usually sparse. Visit their set bits rather
        // than every physical/generator column. A pivot row has no earlier
        // bits, so skipping its preceding words preserves echelon reduction.
        for (size_t w = 0; w < row.size(); w++) {
            uint64_t pending = row[w];
            while (pending) {
                unsigned offset = std::countr_zero(pending);
                size_t p = w * 64 + offset;
                if (p >= width_) break;
                if (pivots_[p].empty()) {
                    pending &= pending - 1;
                    continue;
                }
                for (size_t k = w; k < row.size(); k++) row[k] ^= pivots_[p][k];
                // Lineage pivots need not have their first bit at the
                // physical pivot, so their entire coefficient row is XORed.
                if (coeff) xor_into(*coeff, lineage_[p]);
                pending = offset == 63 ? 0 : row[w] & (~uint64_t{0} << (offset + 1));
            }
        }
    }
    bool insert(Bits row, Bits lineage, size_t source_index) {
        reduce(row, &lineage);
        auto p = first_bit(row, width_);
        if (p == width_) return false;
        pivots_[p] = std::move(row);
        lineage_[p] = std::move(lineage);
        selected_.push_back(source_index);
        return true;
    }
    bool insert_span(Bits row) {
        Bits empty(words(source_count_));
        return insert(std::move(row), std::move(empty), rank());
    }
    PackedRows residues(const PackedRows &rows) const {
        if (rows.width != width_) throw std::invalid_argument("Row widths differ");
        PackedRows result(rows);
        for (auto &row : result.data) reduce(row);
        return result;
    }
    PackedRows coefficients(const PackedRows &targets, bool require_known = true) const {
        if (targets.width != width_) throw std::invalid_argument("Row widths differ");
        PackedRows result(source_count_);
        for (auto row : targets.data) {
            Bits coeff(words(source_count_));
            reduce(row, &coeff);
            if (nonzero(row)) {
                if (require_known) throw std::invalid_argument("Target lies outside the declared source span");
                std::fill(coeff.begin(), coeff.end(), 0);
            }
            result.data.push_back(std::move(coeff));
        }
        return result;
    }
    py::tuple decompose(const PackedRows &targets) const {
        if (targets.width != width_) throw std::invalid_argument("Row widths differ");
        PackedRows coefficients(source_count_);
        py::array_t<bool> dependent(targets.size());
        Basis new_space(width_, 0);
        Indices added;
        for (size_t i = 0; i < targets.size(); i++) {
            auto residual = targets.data[i];
            Bits coeff(words(source_count_));
            reduce(residual, &coeff);
            bool known = !nonzero(residual);
            dependent.mutable_data()[i] = known;
            if (!known) {
                std::fill(coeff.begin(), coeff.end(), 0);
                if (new_space.insert_span(std::move(residual))) added.push_back(i);
            }
            coefficients.data.push_back(std::move(coeff));
        }
        return py::make_tuple(coefficients.numpy(), dependent, added);
    }
    size_t estimated_bytes() const {
        return width_ * sizeof(Bits) * 2 + rank() * (words(width_) + words(source_count_)) * 8;
    }
};

inline PackedRows multiply(const PackedRows &coeff, const PackedRows &rows) {
    if (coeff.width != rows.size()) throw std::invalid_argument("Coefficient width must equal source row count");
    PackedRows result(rows.width);
    for (const auto &c : coeff.data) {
        Bits value(words(rows.width));
        for (size_t w = 0; w < c.size(); w++) {
            auto b = c[w];
            while (b) {
                size_t i = w * 64 + std::countr_zero(b);
                if (i < rows.size()) xor_into(value, rows.data[i]);
                b &= b - 1;
            }
        }
        result.data.push_back(std::move(value));
    }
    return result;
}

inline PackedRows dependencies(const PackedRows &rows) {
    Basis basis(rows.width, rows.size());
    PackedRows result(rows.size());
    for (size_t i = 0; i < rows.size(); i++) {
        auto residual = rows.data[i];
        auto lineage = unit(rows.size(), i);
        basis.reduce(residual, &lineage);
        if (nonzero(residual)) basis.insert(std::move(residual), std::move(lineage), i);
        else result.data.push_back(std::move(lineage));
    }
    return result;
}

struct Annotated {
    PackedRows rows, lineage;
    Annotated(size_t width, size_t count) : rows(width), lineage(count) {}
    void append(const Bits &row, const Bits &coeff) {
        rows.data.push_back(row);
        lineage.data.push_back(coeff);
    }
};

// Each dependency of source residues is a source-space intersection vector.
// No physical-width nullspace matrix is allocated, and the role basis is
// already reduced once by the caller for all closure branches.
inline Annotated intersection_from_residues(const PackedRows &source, const PackedRows &residues) {
    Annotated result(source.width, source.size());
    auto deps = dependencies(residues);
    auto candidates = multiply(deps, source);
    Basis independent(source.width, 0);
    for (size_t i = 0; i < candidates.size(); i++)
        if (independent.insert_span(candidates.data[i])) result.append(candidates.data[i], deps.data[i]);
    return result;
}

inline Annotated preserving_intersection(const PackedRows &source, const PackedRows &residues) {
    Annotated result(source.width, source.size());
    Basis independent(source.width, 0);
    for (size_t i = 0; i < source.size(); i++) if (!nonzero(residues.data[i])) {
        result.append(source.data[i], unit(source.size(), i));
        independent.insert_span(source.data[i]);
    }
    if (result.rows.size() == source.size()) return result;
    auto all = intersection_from_residues(source, residues);
    for (size_t i = 0; i < all.rows.size(); i++)
        if (independent.insert_span(all.rows.data[i])) result.append(all.rows.data[i], all.lineage.data[i]);
    return result;
}

inline Indices checked_selection(Indices indices, size_t count) {
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    if (!indices.empty() && indices.back() >= count) throw std::out_of_range("Postselected row index out of range");
    return indices;
}

inline py::dict classify(const PackedRows &source, const PackedRows &role, Indices selected) {
    if (source.width != role.width) throw std::invalid_argument("Row widths differ");
    selected = checked_selection(std::move(selected), source.size());
    Basis role_basis(role);
    auto residues = role_basis.residues(source);
    if (std::all_of(residues.data.begin(), residues.data.end(), [](const Bits &r) { return !nonzero(r); })) {
        PackedRows identity(source.size());
        std::map<size_t, size_t> originals, exact;
        for (size_t i = 0; i < source.size(); i++) { identity.data.push_back(unit(source.size(), i)); originals[i] = i; }
        for (size_t i : selected) exact[i] = i;
        py::dict out;
        out["bare"] = source;
        out["dressed"] = PackedRows(source.width);
        out["bare_lineage"] = identity;
        out["dressed_lineage"] = PackedRows(source.size());
        out["original_row_map"] = originals;
        out["exact_postselection_map"] = exact;
        out["safely_postselected_output_rows"] = selected;
        out["dressed_quotient_rank"] = 0;
        return out;
    }
    auto bare = preserving_intersection(source, residues);
    Annotated dressed(source.width, source.size());
    Basis output_span(bare.rows);
    std::map<size_t, size_t> original_map, exact_map;
    size_t original_bare_count = 0;
    for (size_t i = 0; i < source.size(); i++) if (!nonzero(residues.data[i]))
        original_map[i] = original_bare_count++;
    for (size_t i : selected) if (nonzero(residues.data[i])) {
        original_map[i] = bare.rows.size() + dressed.rows.size();
        dressed.append(source.data[i], unit(source.size(), i));
        output_span.insert_span(source.data[i]);
    }
    for (size_t i = 0; i < source.size(); i++) if (output_span.insert_span(source.data[i])) {
        original_map[i] = bare.rows.size() + dressed.rows.size();
        dressed.append(source.data[i], unit(source.size(), i));
    }
    for (size_t i : selected) exact_map[i] = original_map.at(i);
    Bits allowed(words(source.size()));
    for (size_t i : selected) set_bit(allowed, i);
    Indices safe;
    auto output_lineage = bare.lineage;
    output_lineage.append(dressed.lineage);
    for (size_t i = 0; i < output_lineage.size(); i++) {
        bool safe_row = nonzero(output_lineage.data[i]);
        for (size_t w = 0; w < allowed.size(); w++) safe_row &= !(output_lineage.data[i][w] & ~allowed[w]);
        if (safe_row) safe.push_back(i);
    }
    py::dict out;
    out["bare"] = bare.rows;
    out["dressed"] = dressed.rows;
    out["bare_lineage"] = bare.lineage;
    out["dressed_lineage"] = dressed.lineage;
    out["original_row_map"] = original_map;
    out["exact_postselection_map"] = exact_map;
    out["safely_postselected_output_rows"] = safe;
    out["dressed_quotient_rank"] = Basis(residues).rank();
    return out;
}

inline PackedRows expand_lineage(const PackedRows &local, const Indices &indices, size_t total) {
    PackedRows result(total);
    for (const auto &row : local.data) {
        Bits target(words(total));
        for (size_t i = 0; i < indices.size(); i++) if (bit(row, i)) set_bit(target, indices[i]);
        result.data.push_back(std::move(target));
    }
    return result;
}
inline py::dict closure_dict(const Annotated &rows, const Basis &measurement_basis) {
    py::dict out;
    out["rows"] = rows.rows;
    out["state_lineage"] = rows.lineage;
    out["measurement_lineage"] = measurement_basis.coefficients(rows.rows);
    return out;
}
inline py::dict readout(const PackedRows &bare, const PackedRows &dressed, const PackedRows &live,
                        const PackedRows &measurements, Indices selected) {
    auto state = bare;
    state.append(dressed);
    state.append(live);
    selected = checked_selection(std::move(selected), state.size());
    Basis measurement_basis(measurements);
    auto residues = measurement_basis.residues(state);
    Indices bare_indices(bare.size());
    std::iota(bare_indices.begin(), bare_indices.end(), 0);
    auto detectors = preserving_intersection(bare, residues.select(bare_indices));
    detectors.lineage = expand_lineage(detectors.lineage, bare_indices, state.size());
    auto full = intersection_from_residues(state, residues);
    Annotated observables(state.width, state.size());
    Basis closed_span(detectors.rows);
    for (size_t i = bare.size(); i < state.size(); i++)
        if (!nonzero(residues.data[i]) && closed_span.insert_span(state.data[i]))
            observables.append(state.data[i], unit(state.size(), i));
    for (size_t i = 0; i < full.rows.size(); i++)
        if (closed_span.insert_span(full.rows.data[i])) observables.append(full.rows.data[i], full.lineage.data[i]);
    auto postselection = preserving_intersection(state.select(selected), residues.select(selected));
    postselection.lineage = expand_lineage(postselection.lineage, selected, state.size());
    py::dict out;
    out["detectors"] = closure_dict(detectors, measurement_basis);
    out["observables"] = closure_dict(observables, measurement_basis);
    out["postselection"] = closure_dict(postselection, measurement_basis);
    return out;
}

inline py::array_t<uint8_t> nullspace(const Rows &matrix) {
    auto source = PackedRows(matrix);
    PackedRows transpose(source.size());
    transpose.data.resize(source.width, Bits(words(source.size())));
    for (size_t i = 0; i < source.size(); i++)
        for (size_t c = 0; c < source.width; c++) if (bit(source.data[i], c)) set_bit(transpose.data[c], i);
    return dependencies(transpose).numpy();
}

inline std::vector<std::vector<int64_t>> combine_records(
        const PackedRows &coefficients, const std::vector<std::vector<int64_t>> &records) {
    if (coefficients.width != records.size()) throw std::invalid_argument("Coefficient width must equal records count");
    std::vector<std::vector<int64_t>> result;
    for (const auto &row : coefficients.data) {
        std::set<int64_t> parity;
        for (size_t w = 0; w < row.size(); w++) {
            auto bits = row[w];
            while (bits) {
                size_t i = w * 64 + std::countr_zero(bits);
                if (i < records.size()) for (int64_t record : records[i]) {
                    auto found = parity.find(record);
                    if (found == parity.end()) parity.insert(record); else parity.erase(found);
                }
                bits &= bits - 1;
            }
        }
        result.emplace_back(parity.begin(), parity.end());
    }
    return result;
}

inline py::array_t<uint8_t> commutativity(const Rows &left, const Rows &right) {
    if (left.ndim() != 2 || right.ndim() != 2 || left.shape(1) != right.shape(1)
            || left.shape(1) % 2)
        throw std::invalid_argument("Pauli rows must be matrices of the same even width");
    const size_t n = left.shape(1) / 2, count_words = words(n);
    struct Split { std::vector<Bits> x, z; };
    auto split = [=](const Rows &rows) {
        Split out;
        out.x.resize(rows.shape(0), Bits(count_words));
        out.z.resize(rows.shape(0), Bits(count_words));
        for (size_t r = 0; r < (size_t)rows.shape(0); r++) {
            const auto *row = rows.data() + r * 2 * n;
            for (size_t q = 0; q < n; q++) {
                if (row[q] & 1) set_bit(out.x[r], q);
                if (row[n + q] & 1) set_bit(out.z[r], q);
            }
        }
        return out;
    };
    auto a = split(left), b = split(right);
    py::array_t<uint8_t> result({left.shape(0), right.shape(0)});
    for (size_t i = 0; i < a.x.size(); i++)
        for (size_t j = 0; j < b.x.size(); j++) {
            unsigned parity = 0;
            for (size_t w = 0; w < count_words; w++)
                parity ^= std::popcount((a.x[i][w] & b.z[j][w]) ^ (a.z[i][w] & b.x[j][w]));
            result.mutable_data()[i * b.x.size() + j] = parity & 1;
        }
    return result;
}

inline void bind(py::module_ &mod) {
    py::class_<PackedRows>(mod, "PackedRows")
        .def(py::init<const Rows &>())
        .def("numpy", &PackedRows::numpy)
        .def("select", &PackedRows::select)
        .def("combined", [](const PackedRows &self, const PackedRows &other) { auto out = self; out.append(other); return out; })
        .def_property_readonly("width", [](const PackedRows &self) { return self.width; })
        .def("__len__", &PackedRows::size);
    py::class_<Basis>(mod, "PackedBasis")
        .def(py::init<const PackedRows &>())
        .def(py::init([](const Rows &rows) { return Basis(PackedRows(rows)); }))
        .def("extended", [](const Basis &self, const Rows &rows) { return self.extended(PackedRows(rows)); })
        .def("decompose", [](const Basis &self, const Rows &targets) { return self.decompose(PackedRows(targets)); })
        .def("decompose_packed", &Basis::decompose)
        .def("coefficients", &Basis::coefficients, py::arg("targets"), py::arg("require_known") = true)
        .def("residues", &Basis::residues)
        .def_property_readonly("rank", &Basis::rank)
        .def_property_readonly("selected", &Basis::selected)
        .def_property_readonly("estimated_bytes", &Basis::estimated_bytes);
    mod.def("packed_classify_stabilizers", &classify);
    mod.def("packed_readout_closures", &readout);
    mod.def("packed_dependencies", &dependencies);
    mod.def("packed_combine_records", &combine_records);
    mod.def("packed_commutativity", &commutativity);
    mod.def("packed_multiply", &multiply);
    mod.def("packed_nullspace", &nullspace);
    mod.def("packed_independent_rows", [](const Rows &rows) { return Basis(PackedRows(rows)).selected(); });
    mod.def("packed_reduce_modulo", [](const Rows &rows, const Rows &basis) {
        return Basis(PackedRows(basis)).residues(PackedRows(rows)).numpy();
    });
    mod.def("packed_intersect", [](const Rows &left, const Rows &right) {
        auto source = PackedRows(left);
        auto residues = Basis(PackedRows(right)).residues(source);
        auto out = intersection_from_residues(source, residues);
        return py::make_tuple(out.rows.numpy(), out.lineage.numpy());
    });
}
}  // namespace lightstim_packed
