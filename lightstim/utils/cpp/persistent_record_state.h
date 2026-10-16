#pragma once

// Persistent record-carrying inverse-tableau state. The physical frame and
// generator re-basing follow Hou Yuewen (isaachyw)'s PR #110, commit 37df61c.
// Original author: isaachyw <isaachyw@umich.edu>. Apache-2.0.
// Detector/logical classification, projection pivot choice, and history remain
// with the current Python Tracker. Redundant source rows retain their records.

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include "stim/circuit/circuit.h"
#include "stim/gates/gates.h"
#include "stim/simulators/tableau_simulator.h"

class PersistentRecordState {
    using Bits = std::vector<uint64_t>;
    using Record = std::vector<int64_t>;
    size_t n;
    stim::TableauSimulator<W> sim;
    mutable std::unique_ptr<stim::TableauTransposedRaii<W>> transposed;
    std::vector<Bits> coordinates;
    std::vector<Record> row_records;
    // Cached packed physical image is a view, never a second state. Invalidated
    // by evolution/projection; public NumPy exports are independent copies.
    std::vector<Bits> physical_cache;
    bool physical_cache_valid = false;
    Rows batch_rows;
    std::vector<Bits> batch_x, batch_z;
    Indices batch_previous;
    bool batch_active = false;
    bool relation_cache_valid = false;
    size_t relation_ns = 0, relation_nl = 0;
    Indices relation_dressed, relation_bare, relation_known, relation_physical;
    std::map<Bits, size_t> relation_exact;
    std::map<std::string, uint64_t> counters;
    stim::TableauTransposedRaii<W> &begin() {
        if (!transposed) transposed = std::make_unique<stim::TableauTransposedRaii<W>>(sim.inv_state);
        return *transposed;
    }
    void end() const { transposed.reset(); }

    // Cached measurement queries are full symplectic coordinates in this
    // same inverse frame. Rebase them together with source Z coordinates;
    // each physical measurement is transformed only once per atomic batch.
    void batch_zcx(size_t src, size_t dst) {
        if (!batch_active) return;
        for (size_t i = 0; i < batch_x.size(); i++) {
            if (bit(batch_x[i], src)) flip(batch_x[i], dst);
            if (bit(batch_z[i], dst)) flip(batch_z[i], src);
        }
    }
    void batch_zcz(size_t a, size_t b) {
        if (!batch_active) return;
        for (size_t i = 0; i < batch_x.size(); i++) {
            if (bit(batch_x[i], a)) flip(batch_z[i], b);
            if (bit(batch_x[i], b)) flip(batch_z[i], a);
        }
    }
    void batch_hxz(size_t q) {
        if (!batch_active) return;
        for (size_t i = 0; i < batch_x.size(); i++)
            if (bit(batch_x[i], q) != bit(batch_z[i], q)) {
                flip(batch_x[i], q); flip(batch_z[i], q);
            }
    }
    void batch_hyz(size_t q) {
        if (!batch_active) return;
        for (size_t i = 0; i < batch_x.size(); i++)
            if (bit(batch_z[i], q)) flip(batch_x[i], q);
    }

    static bool bit(const Bits &a, size_t k) { return (a[k / 64] >> (k % 64)) & 1; }
    static void flip(Bits &a, size_t k) { a[k / 64] ^= uint64_t{1} << (k % 64); }
    static void xorin(Bits &a, const Bits &b) {
        for (size_t k = 0; k < a.size(); k++) a[k] ^= b[k];
    }
    static bool dot(const Bits &a, const Bits &b) {
        unsigned parity = 0;
        for (size_t k = 0; k < a.size(); k++) parity ^= std::popcount(a[k] & b[k]);
        return parity & 1;
    }
    static void xor_records(Record &a, const Record &b, bool preserve_unknown = false) {
        // Projection/folding match PauliTableau's set-based row update. Public
        // record lists themselves are preserved until an operation updates a
        // row; import and Clifford propagation must not rewrite those lists.
        Record left = normalized(a), right = normalized(b);
        bool unknown = preserve_unknown &&
            ((!left.empty() && left.front() < 0) || (!right.empty() && right.front() < 0));
        Record out;
        std::set_symmetric_difference(left.begin(), left.end(), right.begin(), right.end(),
                                      std::back_inserter(out));
        if (unknown && (out.empty() || out.front() >= 0)) out.insert(out.begin(), -1);
        a.swap(out);
    }
    static Record normalized(Record r) {
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        return r;
    }
    void validate_rows(const Rows &rows) const {
        if (rows.ndim() != 2 || (size_t)rows.shape(1) != 2 * n)
            throw std::invalid_argument("Rows must have shape (M, 2*num_qubits)");
    }
    const uint8_t *validate_target(const Rows &target) const {
        if (!((target.ndim() == 1 && (size_t)target.shape(0) == 2 * n) ||
              (target.ndim() == 2 && target.shape(0) == 1 && (size_t)target.shape(1) == 2 * n)))
            throw std::invalid_argument("Target must be one Pauli row");
        return target.data();
    }

    Bits join_coordinates(const Bits &a, const Bits &b) const {
        Bits row((2 * n + 63) / 64, 0);
        std::copy(a.begin(), a.end(), row.begin());
        size_t offset = n / 64, shift = n % 64;
        for (size_t w = 0; w < b.size(); w++) {
            row[offset + w] |= b[w] << shift;
            if (shift && offset + w + 1 < row.size())
                row[offset + w + 1] |= b[w] >> (64 - shift);
        }
        return row;
    }

    // Bulk input conversion reads normal inverse rows as contiguous words.
    // XORing the images of the physical factors costs O(weight*n/64), whereas
    // extracting each generator bit from a transposed frame costs O(weight*n).
    // Call end() ONCE before a whole batch, never once per individual query.
    void query_coordinates_normal(const uint8_t *row, Bits &a, Bits &b) const {
        a.assign((n + 63) / 64, 0); b.assign((n + 63) / 64, 0);
        const auto &inv = sim.inv_state;
        for (size_t q = 0; q < n; q++) {
            if (row[q] & 1) {
                const auto image = inv.xs[q];
                for (size_t w = 0; w < a.size(); w++) {
                    a[w] ^= image.xs.u64[w]; b[w] ^= image.zs.u64[w];
                }
            }
            if (row[n + q] & 1) {
                const auto image = inv.zs[q];
                for (size_t w = 0; w < a.size(); w++) {
                    a[w] ^= image.xs.u64[w]; b[w] ^= image.zs.u64[w];
                }
            }
        }
        if (n % 64) {
            uint64_t mask = (uint64_t{1} << (n % 64)) - 1;
            a.back() &= mask; b.back() &= mask;
        }
    }

    // Valid under TableauTransposedRaii: C^-1 P C = X^a Z^b.
    void query_coordinates(const uint8_t *row, Bits &a, Bits &b) const {
        a.assign((n + 63) / 64, 0);
        b.assign((n + 63) / 64, 0);
        Indices x, z;
        for (size_t q = 0; q < n; q++) {
            if (row[q] & 1) x.push_back(q);
            if (row[n + q] & 1) z.push_back(q);
        }
        const auto &inv = sim.inv_state;
        // Dense role rows and imported combinations should use word parity,
        // not one random bit load for each physical factor and generator.
        if (x.size() + z.size() > 2 * ((n + 63) / 64)) {
            Bits px((n + 63) / 64), pz((n + 63) / 64);
            for (size_t q : x) flip(px, q);
            for (size_t q : z) flip(pz, q);
            for (size_t k = 0; k < n; k++) {
                unsigned ax = 0, bz = 0;
                for (size_t w = 0; w < px.size(); w++) {
                    ax ^= std::popcount((px[w] & inv.xs.xt[k].u64[w]) ^
                                        (pz[w] & inv.zs.xt[k].u64[w]));
                    bz ^= std::popcount((px[w] & inv.xs.zt[k].u64[w]) ^
                                        (pz[w] & inv.zs.zt[k].u64[w]));
                }
                if (ax & 1) flip(a, k);
                if (bz & 1) flip(b, k);
            }
            return;
        }
        for (size_t k = 0; k < n; k++) {
            bool ax = false, bz = false;
            for (size_t q : x) { ax ^= inv.xs.xt[k][q]; bz ^= inv.xs.zt[k][q]; }
            for (size_t q : z) { ax ^= inv.zs.xt[k][q]; bz ^= inv.zs.zt[k][q]; }
            if (ax) flip(a, k);
            if (bz) flip(b, k);
        }
    }
    void cache_physical() {
        if (physical_cache_valid) return;
        end();
        auto forward = sim.inv_state.inverse(true);
        const size_t words = (2 * n + 63) / 64;
        std::vector<Bits> packed(n, Bits(words, 0));
        for (size_t k = 0; k < n; k++) {
            auto z = forward.zs[k];
            for (size_t q = 0; q < n; q++) {
                if (z.xs[q]) flip(packed[k], q);
                if (z.zs[q]) flip(packed[k], n + q);
            }
        }
        physical_cache.assign(coordinates.size(), Bits(words, 0));
        for (size_t r = 0; r < coordinates.size(); r++)
            for (size_t k = 0; k < n; k++)
                if (bit(coordinates[r], k)) xorin(physical_cache[r], packed[k]);
        physical_cache_valid = true;
        counters["physical_materializations"]++;
    }
    Record combine_record_lineage(const Bits &coeff, const std::vector<Record> &records) const {
        std::set<int64_t> parity;
        bool unknown = false;
        for (size_t i = 0; i < records.size(); i++) if (bit(coeff, i))
            for (int64_t r : records[i]) {
                unknown |= r < 0;
                if (parity.count(r)) parity.erase(r); else parity.insert(r);
            }
        if (unknown && (parity.empty() || *parity.begin() >= 0)) parity.insert(-1);
        return Record(parity.begin(), parity.end());
    }
    void anchor(stim::TableauTransposedRaii<W> &trans, size_t g,
                const Bits &a, const Bits &b) {
        if (bit(a, g)) {
            for (size_t k = 0; k < n; k++)
                if (k != g && bit(b, k)) { trans.append_ZCZ(g, k); batch_zcz(g, k); counters["frame_rebases"]++; }
            if (bit(b, g)) { trans.append_H_YZ(g); batch_hyz(g); }
            else { trans.append_H_XZ(g); batch_hxz(g); }
            counters["frame_rebases"]++;
        } else {
            for (size_t k = 0; k < n; k++)
                if (k != g && bit(b, k)) { trans.append_ZCX(k, g); batch_zcx(k, g); counters["frame_rebases"]++; }
        }
    }
    // Change S_dst to S_dst*S_src and update coordinates of every retained row.
    void multiply_generator(stim::TableauTransposedRaii<W> &trans, size_t src, size_t dst) {
        trans.append_ZCX(src, dst);
        batch_zcx(src, dst);
        for (auto &c : coordinates) if (bit(c, dst)) flip(c, src);
        counters["frame_rebases"]++;
    }
    struct Solver {
        std::vector<Bits> pivots, coeffs;
    };
    mutable std::map<Indices, Solver> solver_cache;
    Solver make_solver(const Indices &indices) const {
        Solver s{std::vector<Bits>(n), std::vector<Bits>(n)};
        for (size_t j = 0; j < indices.size(); j++) {
            size_t source = indices[j];
            if (source >= coordinates.size()) throw std::out_of_range("Source row index out of range");
            Bits r = coordinates[source], c((coordinates.size() + 63) / 64, 0);
            flip(c, source);
            for (size_t p = 0; p < n; p++) {
                if (!bit(r, p)) continue;
                if (s.pivots[p].empty()) {
                    s.pivots[p] = std::move(r); s.coeffs[p] = std::move(c); break;
                }
                xorin(r, s.pivots[p]); xorin(c, s.coeffs[p]);
            }
        }
        return s;
    }
    struct Decomposition {
        bool represented = false;
        Bits selected;
        Record parity;
    };
    struct Relation {
        std::string kind;
        Record parity;
        Indices logical;
        py::tuple python() const { return py::make_tuple(kind, parity, logical); }
    };
    Decomposition decompose_coordinates_native(const Bits &a, Bits b, const Indices &indices) const {
        Decomposition result{false, Bits((coordinates.size() + 63) / 64, 0), {}};
        for (uint64_t w : a) if (w) return result;
        auto entry = solver_cache.find(indices);
        if (entry == solver_cache.end()) {
            if (solver_cache.size() >= 8) solver_cache.clear();
            entry = solver_cache.emplace(indices, make_solver(indices)).first;
        }
        const auto &solver = entry->second;
        for (size_t p = 0; p < n; p++) {
            if (!bit(b, p)) continue;
            if (solver.pivots[p].empty()) {
                std::fill(result.selected.begin(), result.selected.end(), uint64_t{0});
                return result;
            }
            xorin(b, solver.pivots[p]); xorin(result.selected, solver.coeffs[p]);
        }
        Record occurrences;
        bool unknown = false;
        for (size_t k = 0; k < coordinates.size(); k++) if (bit(result.selected, k)) {
            occurrences.insert(occurrences.end(), row_records[k].begin(), row_records[k].end());
            for (int64_t r : row_records[k]) unknown |= r < 0;
        }
        // A linear combination follows combine_records: each occurrence
        // toggles parity, including duplicates within a public source list.
        std::sort(occurrences.begin(), occurrences.end());
        Record record;
        for (size_t i = 0; i < occurrences.size();) {
            size_t j = i + 1;
            while (j < occurrences.size() && occurrences[j] == occurrences[i]) j++;
            if ((j - i) & 1) record.push_back(occurrences[i]);
            i = j;
        }
        if (unknown && (record.empty() || record.front() >= 0)) record.insert(record.begin(), -1);
        result.represented = true;
        result.parity = std::move(record);
        return result;
    }
    py::tuple decompose_coordinates(const Bits &a, Bits b, const Indices &indices) const {
        auto result = decompose_coordinates_native(a, std::move(b), indices);
        py::array_t<uint8_t> coeff(coordinates.size());
        for (size_t i = 0; i < coordinates.size(); i++) coeff.mutable_data()[i] = bit(result.selected, i);
        return py::make_tuple(coeff, result.represented, result.parity);
    }
public:
    PersistentRecordState(const Rows &rows, const std::vector<Record> &records)
        : n(rows.ndim() == 2 ? rows.shape(1) / 2 : 0), sim(std::mt19937_64(0), n) {
        if (rows.ndim() != 2 || rows.shape(1) % 2)
            throw std::invalid_argument("Rows must be a 2D even-width matrix");
        for (const auto *name : {"constructions", "replacements", "unitaries", "queries",
             "projections", "folds", "frame_rebases", "snapshots", "decompositions"}) counters[name] = 0;
        counters["fold_fastpaths"] = 0;
        counters["constructions"] = 1;
        replace_state(rows, records);
    }
    PersistentRecordState(const PersistentRecordState &other)
        : n(other.n), sim((other.end(), other.sim), std::mt19937_64(0)), coordinates(other.coordinates),
          row_records(other.row_records), counters(other.counters) {}

    void replace_state(const Rows &rows, const std::vector<Record> &records) {
        validate_rows(rows);
        if (records.size() != (size_t)rows.shape(0)) throw std::invalid_argument("Rows and records must have equal length");
        counters["replacements"]++;
        relation_cache_valid = false;
        // Reconcile writable public views without rebuilding the inverse
        // frame. Row selection/reordering and arbitrary products of the old
        // state stay in the same generator coordinates.
        std::map<Bits, Bits> old_rows;
        if (physical_cache_valid)
            for (size_t i = 0; i < coordinates.size(); i++)
                old_rows.emplace(physical_cache[i], coordinates[i]);
        std::vector<Bits> next, next_physical;
        bool same_frame = true;
        end();
        for (py::ssize_t r = 0; r < rows.shape(0); r++) {
            Bits physical((2 * n + 63) / 64, 0);
            const auto *row = rows.data() + r * 2 * n;
            for (size_t p = 0; p < 2 * n; p++) if (row[p] & 1) flip(physical, p);
            auto match = old_rows.find(physical);
            if (match != old_rows.end()) {
                next.push_back(match->second);
                counters["reused_coordinate_rows"]++;
            } else {
                Bits a, b;
                query_coordinates_normal(row, a, b);
                if (std::any_of(a.begin(), a.end(), [](uint64_t w) { return w != 0; })) {
                    same_frame = false;
                    break;
                }
                next.push_back(std::move(b));
                counters["coordinate_import_rows"]++;
            }
            next_physical.push_back(std::move(physical));
        }
        if (same_frame) {
            if (next == coordinates && records == row_records) counters["reconcile_unchanged"]++;
            else solver_cache.clear();
            coordinates = std::move(next);
            row_records = records;
            physical_cache = std::move(next_physical);
            physical_cache_valid = true;
            return;
        }
        physical_cache_valid = false;
        solver_cache.clear();
        coordinates.clear(); row_records.clear();
        std::vector<uint8_t> known(n, 0);
        auto &trans = begin();
        Bits a, b;
        for (py::ssize_t r = 0; r < rows.shape(0); r++) {
            query_coordinates(rows.data() + r * 2 * n, a, b);
            size_t g = n;
            for (size_t k = 0; k < n; k++) if (bit(a, k)) {
                if (known[k]) throw std::invalid_argument("PersistentRecordState requires mutually commuting source rows");
                if (g == n) g = k;
            }
            if (g != n) {
                for (size_t k = 0; k < n; k++) if (k != g && bit(a, k)) {
                    trans.append_ZCX(g, k); batch_zcx(g, k); counters["frame_rebases"]++;
                }
                query_coordinates(rows.data() + r * 2 * n, a, b);
            } else {
                for (size_t k = 0; k < n; k++) if (bit(b, k) && !known[k]) { g = k; break; }
            }
            if (g != n) { anchor(trans, g, a, b); known[g] = 1; }
        }
        coordinates.reserve(records.size()); row_records.reserve(records.size());
        for (py::ssize_t r = 0; r < rows.shape(0); r++) {
            query_coordinates(rows.data() + r * 2 * n, a, b);
            coordinates.push_back(std::move(b));
            row_records.push_back(records[r]);
        }
    }
    py::tuple query(const Rows &target) {
        counters["queries"]++;
        const auto *row = validate_target(target);
        Bits a, b;
        begin(); query_coordinates(row, a, b);
        Indices anti, all;
        for (size_t k = 0; k < coordinates.size(); k++) {
            all.push_back(k);
            if (dot(a, coordinates[k])) anti.push_back(k);
        }
        auto d = decompose_coordinates(a, b, all);
        return py::make_tuple(anti, d[0], d[1], d[2]);
    }
    Indices anticommuting(const Rows &target) {
        counters["queries"]++;
        const auto *row = validate_target(target);
        Bits a, b;
        begin(); query_coordinates(row, a, b);
        Indices anti;
        for (size_t k = 0; k < coordinates.size(); k++)
            if (dot(a, coordinates[k])) anti.push_back(k);
        return anti;
    }
    py::tuple decompose(const Rows &target, const Indices &indices) {
        counters["decompositions"]++;
        const auto *row = validate_target(target);
        Bits a, b;
        begin(); query_coordinates(row, a, b);
        return decompose_coordinates(a, b, indices);
    }
    py::tuple commuting_relation(const Rows &target, size_t ns, size_t nl,
                                  const Indices &dressed, const Rows &measurements,
                                  size_t index, int64_t base_record) {
        return commuting_relation_native(validate_target(target), ns, nl, dressed,
                                          measurements, index, base_record, false).python();
    }
    Relation commuting_relation_native(const uint8_t *row, size_t ns, size_t nl,
                                       const Indices &dressed, const Rows &measurements,
                                       size_t index, int64_t base_record, bool from_batch) {
        using namespace lightstim_packed;
        validate_rows(measurements);
        if (ns + nl > coordinates.size() || index >= (size_t)measurements.shape(0))
            throw std::out_of_range("Relation source range is invalid");
        if (!relation_cache_valid || relation_ns != ns || relation_nl != nl || relation_dressed != dressed) {
            std::set<size_t> dressed_set(dressed.begin(), dressed.end());
            relation_bare.clear(); relation_known.clear(); relation_physical.clear(); relation_exact.clear();
            for (size_t i = 0; i < ns + nl; i++) {
                relation_physical.push_back(i);
                bool is_known = std::all_of(row_records[i].begin(), row_records[i].end(),
                                           [](int64_t r) { return r >= 0; });
                if (is_known) {
                    relation_known.push_back(i);
                    if (i < ns && !dressed_set.count(i)) {
                        relation_bare.push_back(i);
                        relation_exact.emplace(coordinates[i], i);
                    }
                }
            }
            relation_ns = ns; relation_nl = nl; relation_dressed = dressed;
            relation_cache_valid = true;
            counters["relation_contexts"]++;
        }
        const auto &bare = relation_bare;
        const auto &known = relation_known;
        const auto &physical = relation_physical;
        Bits a, b;
        if (from_batch) { a = batch_x.at(index); b = batch_z.at(index); }
        else { begin(); query_coordinates(row, a, b); }
        bool frame_z = !std::any_of(a.begin(), a.end(), [](uint64_t w) { return w != 0; });
        // Existing local-check representatives have priority over every
        // alternative decomposition, including earlier identical measurements.
        auto exact = relation_exact.find(b);
        if (frame_z && exact != relation_exact.end()) {
            size_t i = exact->second;
            Record parity = normalized(row_records[i]);
            xor_records(parity, Record{base_record + (int64_t)index});
            counters["exact_relations"]++;
            return {"detector", std::move(parity), {}};
        }
        for (size_t j = (from_batch ? batch_previous.at(index) : 0); j < index; j++)
            if (from_batch || std::equal(row, row + 2 * n, measurements.data() + j * 2 * n))
                return {"detector", Record{base_record + (int64_t)j,
                                      base_record + (int64_t)index}, {}};
        auto bare_rows = coordinate_rows(bare);
        if (from_batch) {
            for (size_t j = 0; j < index; j++) {
                bare_rows.data.push_back(join_coordinates(batch_x[j], batch_z[j]));
            }
        } else {
            Rows prefix({(py::ssize_t)index, (py::ssize_t)(2 * n)});
            std::copy_n(measurements.data(), index * 2 * n, prefix.mutable_data());
            bare_rows.append(coordinate_paulis(prefix));
        }
        PackedRows query_row(2 * n);
        query_row.data.push_back(join_coordinates(a, b));
        Basis bare_basis(bare_rows);
        if (!nonzero(bare_basis.residues(query_row).data[0])) {
            auto coeff = bare_basis.coefficients(query_row);
            auto records = records_selected(bare);
            for (size_t j = 0; j < index; j++) records.push_back(Record{base_record + (int64_t)j});
            auto parity = combine_record_lineage(coeff.data[0], records);
            xor_records(parity, Record{base_record + (int64_t)index});
            counters["prefix_decompositions"]++;
            return {"detector", std::move(parity), {}};
        }
        auto result = decompose_coordinates_native(a, b, known);
        counters["decompositions"]++;
        if (!result.represented) {
            auto potential = decompose_coordinates_native(a, b, physical);
            if (potential.represented) {
                for (size_t i = ns; i < ns + nl; i++)
                    if (bit(potential.selected, i) && std::any_of(row_records[i].begin(), row_records[i].end(),
                                                     [](int64_t r) { return r < 0; }))
                        throw std::invalid_argument("Cannot capture logical history from an unmeasured logical record");
            }
            return {"unknown", {}, {}};
        }
        xor_records(result.parity, Record{base_record + (int64_t)index});
        Indices logical;
        for (size_t i = ns; i < ns + nl; i++) if (bit(result.selected, i)) logical.push_back(i - ns);
        return {"logical", std::move(result.parity), std::move(logical)};
    }
    void begin_measurement_batch(const Rows &measurements) {
        validate_rows(measurements);
        batch_active = false;
        batch_rows = Rows({measurements.shape(0), measurements.shape(1)});
        std::copy_n(measurements.data(), measurements.size(), batch_rows.mutable_data());
        batch_x.clear(); batch_z.clear(); batch_previous.clear();
        std::map<std::string, size_t> first;
        end();
        for (py::ssize_t i = 0; i < measurements.shape(0); i++) {
            Bits a, b;
            const auto *row = measurements.data() + i * 2 * n;
            query_coordinates_normal(row, a, b);
            batch_x.push_back(std::move(a)); batch_z.push_back(std::move(b));
            std::string key(reinterpret_cast<const char *>(row), 2 * n);
            auto entry = first.emplace(key, i);
            batch_previous.push_back(entry.second ? i : entry.first->second);
        }
        batch_active = true;
        relation_cache_valid = false;
        counters["measurement_batches"]++;
        counters["measurement_coordinate_imports"] += measurements.shape(0);
    }
    Indices measurement_anticommuting(size_t index) {
        if (!batch_active || index >= batch_x.size()) throw std::out_of_range("No active measurement batch at this index");
        Indices anti;
        for (size_t k = 0; k < coordinates.size(); k++)
            if (dot(batch_x[index], coordinates[k])) anti.push_back(k);
        counters["queries"]++;
        counters["cached_measurement_queries"]++;
        return anti;
    }
    py::tuple measurement_relation(size_t index, size_t ns, size_t nl,
                                    const Indices &dressed, int64_t base_record) {
        if (!batch_active || index >= batch_x.size()) throw std::out_of_range("No active measurement batch at this index");
        return commuting_relation_native(batch_rows.data() + index * 2 * n, ns, nl,
                                          dressed, batch_rows, index, base_record, true).python();
    }
    py::dict process_measurement_batch(const Rows &measurements, size_t ns, size_t nl,
                                      const Indices &dressed, const Indices &classified,
                                      int64_t base_record) {
        if (ns + nl > coordinates.size()) throw std::out_of_range("Measurement source range is invalid");
        begin_measurement_batch(measurements);
        std::set<size_t> classified_rows(classified.begin(), classified.end());
        std::vector<std::pair<size_t, Relation>> events;
        std::string error_type, error_message;
        counters["native_measurement_batches"]++;
        for (size_t i = 0; i < batch_x.size(); i++) {
            const auto *row = batch_rows.data() + i * 2 * n;
            try {
                auto anti = measurement_anticommuting(i);
                if (!anti.empty()) {
                    // Source rows are ordered S, L, L_absorbed. A redundant
                    // S row remains an eligible pivot before every L row.
                    size_t pivot = anti.front();
                    if (pivot >= ns + nl)
                        throw std::runtime_error("L_absorbed cannot be an independent projection pivot");
                    project_row(row, pivot, Record{base_record + (int64_t)i});
                    if (pivot >= ns) {
                        for (size_t j : anti) if (j != pivot) classified_rows.erase(j);
                    } else {
                        classified_rows.insert(pivot);
                    }
                    continue;
                }
                if (std::none_of(row, row + 2 * n, [](uint8_t v) { return v != 0; })) {
                    events.push_back({i, {"detector", Record{base_record + (int64_t)i}, {}}});
                    continue;
                }
                auto relation = commuting_relation_native(row, ns, nl, dressed,
                                                           batch_rows, i, base_record, true);
                if (relation.kind != "unknown") events.emplace_back(i, std::move(relation));
            } catch (const std::invalid_argument &error) {
                error_type = "ValueError"; error_message = error.what(); break;
            } catch (const std::out_of_range &error) {
                error_type = "IndexError"; error_message = error.what(); break;
            } catch (const std::runtime_error &error) {
                error_type = "RuntimeError"; error_message = error.what(); break;
            }
        }
        // Python must replay earlier circuit/history events before raising a
        // later semantic error. The caller operates on a transaction clone,
        // and commits neither this frame nor classified rows on failure.
        py::list output;
        for (const auto &[index, relation] : events)
            output.append(py::make_tuple(index, relation.kind, relation.parity, relation.logical));
        py::dict result;
        result["events"] = std::move(output);
        result["classified"] = Indices(classified_rows.begin(), classified_rows.end());
        result["error"] = py::none();
        if (!error_type.empty()) result["error"] = py::make_tuple(error_type, error_message);
        return result;
    }
    Indices project_readout_batch(const Rows &measurements, const Record &records,
                                 size_t physical_end) {
        validate_rows(measurements);
        if (records.size() != (size_t)measurements.shape(0))
            throw std::invalid_argument("Readout record count differs from measurements");
        if (physical_end > coordinates.size()) throw std::out_of_range("Readout source range is invalid");
        begin_measurement_batch(measurements);
        Indices destroyed;
        counters["native_readout_batches"]++;
        for (size_t i = 0; i < batch_x.size(); i++) {
            auto anti = measurement_anticommuting(i);
            auto pivot = std::find_if(anti.begin(), anti.end(), [&](size_t row) {
                return !std::binary_search(destroyed.begin(), destroyed.end(), row);
            });
            if (pivot == anti.end()) continue;
            if (*pivot >= physical_end)
                throw std::runtime_error("L_absorbed supplied the only anticommuting readout pivot");
            project_row(batch_rows.data() + i * 2 * n, *pivot, Record{records[i]}, destroyed);
            destroyed.insert(std::lower_bound(destroyed.begin(), destroyed.end(), *pivot), *pivot);
        }
        return destroyed;
    }
    Indices refresh_complement(const Rows &measurements, size_t ns, size_t nl) {
        using namespace lightstim_packed;
        validate_rows(measurements);
        if (ns + nl > coordinates.size()) throw std::out_of_range("Refresh source range is invalid");
        relation_cache_valid = false;
        // The public convention canonicalizes L in physical-column RREF.
        // Operate on packed native physical rows and apply exactly the same
        // lineage to inverse coordinates and records, without a NumPy roundtrip.
        if (nl) {
            cache_physical();
            std::vector<Bits> physical, coeffs;
            for (size_t i = 0; i < nl; i++) {
                physical.push_back(physical_cache[ns + i]);
                coeffs.push_back(unit(nl, i));
            }
            size_t lead = 0;
            for (size_t col = 0; col < 2 * n && lead < nl; col++) {
                size_t pivot = lead;
                while (pivot < nl && !bit(physical[pivot], col)) pivot++;
                if (pivot == nl) continue;
                std::swap(physical[lead], physical[pivot]);
                std::swap(coeffs[lead], coeffs[pivot]);
                for (size_t r = 0; r < nl; r++) if (r != lead && bit(physical[r], col)) {
                    xorin(physical[r], physical[lead]); xorin(coeffs[r], coeffs[lead]);
                }
                lead++;
            }
            auto old_coordinates = coordinates;
            std::vector<Record> old_records(row_records.begin() + ns, row_records.begin() + ns + nl);
            for (size_t r = 0; r < nl; r++) {
                coordinates[ns + r].assign((n + 63) / 64, 0);
                for (size_t i = 0; i < nl; i++) if (bit(coeffs[r], i))
                    xorin(coordinates[ns + r], old_coordinates[ns + i]);
                row_records[ns + r] = combine_record_lineage(coeffs[r], old_records);
                physical_cache[ns + r] = physical[r];
            }
            solver_cache.clear();
        }
        Indices targets;
        for (size_t i = ns; i < ns + nl; i++) targets.push_back(i);
        for (size_t i = 0; i < ns; i++) targets.push_back(i);
        auto base = coordinate_paulis(measurements);
        auto candidate = base;
        candidate.append(coordinate_rows(targets));
        Indices complement;
        Basis candidate_basis(candidate);
        for (size_t i : candidate_basis.selected())
            if (i >= base.size()) complement.push_back(i - base.size());
        // Historical solve_linear_decomposition groups NEW rows by physical
        // support components. Retain that ordering even though rank and
        // coefficients are now solved entirely in inverse coordinates.
        if (complement.size() > 1 && base.size()) {
            cache_physical();
            PackedRows physical(measurements);
            for (size_t i : targets) physical.data.push_back(physical_cache[i]);
            std::vector<size_t> parent(physical.size());
            std::iota(parent.begin(), parent.end(), 0);
            auto find = [&](size_t i) { while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
            std::vector<size_t> first(2 * n, physical.size());
            for (size_t i = 0; i < physical.size(); i++) for (size_t w = 0; w < physical.data[i].size(); w++) {
                auto value = physical.data[i][w];
                while (value) {
                    size_t col = w * 64 + std::countr_zero(value);
                    if (first[col] == physical.size()) first[col] = i;
                    else parent[find(i)] = find(first[col]);
                    value &= value - 1;
                }
            }
            std::map<size_t, size_t> groups;
            std::vector<size_t> component(physical.size());
            for (size_t i = 0; i < physical.size(); i++) {
                auto root = find(i);
                if (!groups.count(root)) groups[root] = groups.size();
                component[i] = groups[root];
            }
            std::stable_sort(complement.begin(), complement.end(), [&](size_t a, size_t b) {
                return component[base.size() + a] < component[base.size() + b];
            });
        }
        counters["native_refreshes"]++;
        return complement;
    }
    void refresh_sources(const Rows &prefix, const std::vector<Record> &prefix_records,
                         const Indices &retained) {
        validate_rows(prefix);
        if (prefix_records.size() != (size_t)prefix.shape(0)) throw std::invalid_argument("Prefix record count differs");
        cache_physical();
        Rows output({(py::ssize_t)(prefix_records.size() + retained.size()), (py::ssize_t)(2 * n)});
        std::copy_n(prefix.data(), prefix_records.size() * 2 * n, output.mutable_data());
        auto records = prefix_records;
        for (size_t r = 0; r < retained.size(); r++) {
            size_t i = retained[r];
            if (i >= coordinates.size()) throw std::out_of_range("Retained source index out of range");
            for (size_t col = 0; col < 2 * n; col++)
                output.mutable_data()[(prefix_records.size() + r) * 2 * n + col] = bit(physical_cache[i], col);
            records.push_back(row_records[i]);
        }
        batch_active = false;
        replace_state(output, records);
        counters["source_refreshes"]++;
    }
    Indices nonzero_rows(const Indices &indices) const {
        Indices out;
        for (size_t i : indices) if (lightstim_packed::nonzero(coordinates.at(i))) out.push_back(i);
        return out;
    }
    Indices ordered_complement(const Indices &prefix, const Indices &targets) {
        using namespace lightstim_packed;
        auto combined = coordinate_rows(prefix);
        combined.append(coordinate_rows(targets));
        Indices result;
        Basis combined_basis(combined);
        for (size_t i : combined_basis.selected())
            if (i >= prefix.size()) result.push_back(i - prefix.size());
        if (prefix.empty() || result.size() < 2) return result;
        cache_physical();
        auto indices = prefix;
        indices.insert(indices.end(), targets.begin(), targets.end());
        std::vector<size_t> parent(indices.size());
        std::iota(parent.begin(), parent.end(), 0);
        auto find = [&](size_t i) { while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
        std::vector<size_t> first(2 * n, indices.size());
        for (size_t i = 0; i < indices.size(); i++) for (size_t w = 0; w < physical_cache[indices[i]].size(); w++) {
            auto value = physical_cache[indices[i]][w];
            while (value) {
                size_t col = w * 64 + std::countr_zero(value);
                if (first[col] == indices.size()) first[col] = i;
                else parent[find(i)] = find(first[col]);
                value &= value - 1;
            }
        }
        std::map<size_t, size_t> groups;
        std::vector<size_t> component(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            auto root = find(i);
            if (!groups.count(root)) groups[root] = groups.size();
            component[i] = groups[root];
        }
        std::stable_sort(result.begin(), result.end(), [&](size_t a, size_t b) {
            return component[prefix.size() + a] < component[prefix.size() + b];
        });
        return result;
    }
    py::tuple trace_qubits(const Indices &qubits, size_t ns, size_t nl) {
        if (ns + nl > coordinates.size()) throw std::out_of_range("Trace source range is invalid");
        relation_cache_valid = false;
        cache_physical();
        Indices columns;
        for (size_t q : qubits) {
            if (q >= n) throw std::out_of_range("Traced qubit exceeds state register");
            columns.push_back(q);
        }
        for (size_t q : qubits) columns.push_back(n + q);
        std::set<size_t> removed;
        for (size_t col : columns) {
            Indices affected;
            for (size_t i = 0; i < coordinates.size(); i++)
                if (!removed.count(i) && bit(physical_cache[i], col)) affected.push_back(i);
            if (affected.empty()) continue;
            size_t pivot = affected[0];
            if (pivot >= ns + nl) throw std::runtime_error("L_absorbed cannot be an independent projection pivot");
            for (size_t i : affected) if (i != pivot) {
                xorin(coordinates[i], coordinates[pivot]);
                xorin(physical_cache[i], physical_cache[pivot]);
                xor_records(row_records[i], row_records[pivot], true);
            }
            std::fill(coordinates[pivot].begin(), coordinates[pivot].end(), 0);
            std::fill(physical_cache[pivot].begin(), physical_cache[pivot].end(), 0);
            row_records[pivot].clear();
            removed.insert(pivot);
        }
        Indices kept;
        size_t new_ns = 0, new_nl = 0, new_na = 0;
        std::vector<Bits> next, next_physical;
        std::vector<Record> records;
        for (size_t i = 0; i < coordinates.size(); i++)
            if (!removed.count(i) && lightstim_packed::nonzero(coordinates[i])) {
                kept.push_back(i);
                if (i < ns) new_ns++; else if (i < ns + nl) new_nl++; else new_na++;
                next.push_back(coordinates[i]); next_physical.push_back(physical_cache[i]);
                records.push_back(row_records[i]);
            }
        coordinates = std::move(next); physical_cache = std::move(next_physical);
        row_records = std::move(records); solver_cache.clear();
        counters["partial_traces"]++;
        return py::make_tuple(kept, new_ns, new_nl, new_na);
    }
    void insert_sources(const Rows &added, const std::vector<Record> &records, size_t position) {
        validate_rows(added);
        if (position > coordinates.size() || records.size() != (size_t)added.shape(0))
            throw std::invalid_argument("Inserted source rows have invalid position or record count");
        cache_physical();
        Rows output({(py::ssize_t)(coordinates.size() + records.size()), (py::ssize_t)(2 * n)});
        std::vector<Record> out_records;
        size_t out = 0;
        for (size_t i = 0; i <= coordinates.size(); i++) {
            if (i == position) {
                std::copy_n(added.data(), records.size() * 2 * n, output.mutable_data() + out * 2 * n);
                out += records.size();
                out_records.insert(out_records.end(), records.begin(), records.end());
            }
            if (i == coordinates.size()) break;
            for (size_t col = 0; col < 2 * n; col++)
                output.mutable_data()[out * 2 * n + col] = bit(physical_cache[i], col);
            out++;
            out_records.push_back(row_records[i]);
        }
        replace_state(output, out_records);
        counters["source_insertions"]++;
    }
    py::tuple fold_rows(const Indices &indices, const Rows &measurements,
                        const std::vector<Record> &measurement_records) {
        using namespace lightstim_packed;
        validate_rows(measurements);
        if ((size_t)measurements.shape(0) != measurement_records.size())
            throw std::invalid_argument("Measured row count differs from records");
        if (indices.empty()) return py::make_tuple(PackedRows(2 * n), std::vector<Record>{});
        cache_physical();
        Basis basis{PackedRows(measurements)};
        PackedRows folded(2 * n);
        std::vector<Record> records;
        for (size_t i : indices) {
            auto row = physical_cache.at(i);
            Bits coeff(words(measurement_records.size()));
            basis.reduce(row, &coeff);
            auto record = normalized(row_records.at(i));
            xor_records(record, combine_record_lineage(coeff, measurement_records));
            folded.data.push_back(std::move(row));
            records.push_back(std::move(record));
        }
        counters["native_readout_folds"]++;
        return py::make_tuple(folded, records);
    }
    void adopt_physical(const lightstim_packed::PackedRows &rows, const std::vector<Record> &records) {
        replace_state(rows.numpy(), records);
        counters["packed_physical_adoptions"]++;
    }
    void project(const Rows &target, size_t pivot, const Record &record, const Indices &excluded = {}) {
        project_row(validate_target(target), pivot, record, excluded);
    }
    void project_row(const uint8_t *row, size_t pivot, const Record &record, const Indices &excluded = {}) {
        if (pivot >= coordinates.size()) throw std::out_of_range("Projection pivot out of range");
        counters["projections"]++;
        relation_cache_valid = false;
        physical_cache_valid = false;
        solver_cache.clear();
        auto &trans = begin();
        Bits a, b;
        query_coordinates(row, a, b);
        if (!dot(a, coordinates[pivot])) throw std::invalid_argument("Projection pivot must anticommute with target");
        for (size_t r : excluded) {
            if (r >= coordinates.size()) throw std::out_of_range("Excluded source row out of range");
            if (dot(a, coordinates[r])) throw std::invalid_argument("Excluded source row anticommutes with projection");
        }
        const Bits old_pivot = coordinates[pivot];
        const Record old_record = row_records[pivot];
        Indices anti;
        for (size_t k = 0; k < coordinates.size(); k++) if (dot(a, coordinates[k])) anti.push_back(k);
        size_t g = 0;
        while (!(bit(old_pivot, g) && bit(a, g))) g++;
        for (size_t k = 0; k < n; k++) if (k != g && bit(old_pivot, k)) multiply_generator(trans, k, g);
        query_coordinates(row, a, b);
        for (size_t k = 0; k < n; k++) if (k != g && bit(a, k)) multiply_generator(trans, g, k);
        query_coordinates(row, a, b);
        for (size_t r : anti) if (r != pivot) {
            flip(coordinates[r], g);
            xor_records(row_records[r], old_record, true);
        }
        coordinates[pivot].assign((n + 63) / 64, 0); flip(coordinates[pivot], g);
        row_records[pivot] = record;
        anchor(trans, g, a, b);
    }
    void unitary(const std::string &text) {
        batch_active = false;
        physical_cache_valid = false;
        end();
        stim::Circuit circuit(text);
        circuit.for_each_operation([&](const stim::CircuitInstruction &op) {
            const auto &gate = stim::GATE_DATA[op.gate_type];
            if (!(gate.flags & stim::GATE_IS_UNITARY)) return;
            for (const auto &t : op.targets) {
                if (t.is_classical_bit_target()) throw std::invalid_argument("Classically controlled gates require explicit Tracker handling");
                if (t.has_qubit_value() && t.qubit_value() >= n) throw std::out_of_range("Gate exceeds state register");
            }
            sim.do_gate(op);
            counters["unitaries"]++;
        });
    }
    py::array_t<uint8_t> rows() {
        counters["snapshots"]++;
        cache_physical();
        py::array_t<uint8_t> result({(py::ssize_t)coordinates.size(), (py::ssize_t)(2 * n)});
        for (size_t r = 0; r < coordinates.size(); r++) {
            for (size_t k = 0; k < 2 * n; k++) result.mutable_data()[r * 2 * n + k] = bit(physical_cache[r], k);
        }
        return result;
    }
    py::array_t<uint8_t> rows_selected(const Indices &indices) {
        counters["view_exports"]++;
        py::array_t<uint8_t> result({(py::ssize_t)indices.size(), (py::ssize_t)(2 * n)});
        if (indices.empty()) return result;
        cache_physical();
        for (size_t r = 0; r < indices.size(); r++) {
            if (indices[r] >= coordinates.size()) throw std::out_of_range("Source row index out of range");
            for (size_t k = 0; k < 2 * n; k++)
                result.mutable_data()[r * 2 * n + k] = bit(physical_cache[indices[r]], k);
        }
        return result;
    }
    std::vector<Record> records_selected(const Indices &indices) const {
        std::vector<Record> result;
        for (size_t i : indices) result.push_back(row_records.at(i));
        return result;
    }
    lightstim_packed::PackedRows coordinate_rows(const Indices &indices) const {
        lightstim_packed::PackedRows out(2 * n);
        for (size_t i : indices) {
            const auto &c = coordinates.at(i);
            out.data.push_back(join_coordinates(Bits{}, c));
        }
        return out;
    }
    lightstim_packed::PackedRows coordinate_paulis(const Rows &rows) {
        validate_rows(rows);
        lightstim_packed::PackedRows out(2 * n);
        end();
        for (py::ssize_t i = 0; i < rows.shape(0); i++) {
            Bits a, b;
            query_coordinates_normal(rows.data() + i * 2 * n, a, b);
            out.data.push_back(join_coordinates(a, b));
        }
        counters["role_coordinate_rows"] += rows.shape(0);
        return out;
    }
    void adopt_coordinates(const lightstim_packed::PackedRows &rows, const std::vector<Record> &records) {
        if (rows.width != 2 * n || rows.size() != records.size())
            throw std::invalid_argument("Coordinate rows and records have incompatible shape");
        std::vector<Bits> next;
        for (const auto &row : rows.data) {
            Bits c((n + 63) / 64, 0);
            for (size_t k = 0; k < n; k++) {
                if (bit(row, k)) throw std::invalid_argument("Adopted source rows must lie in the inverse Z frame");
                if (bit(row, n + k)) flip(c, k);
            }
            next.push_back(std::move(c));
        }
        relation_cache_valid = false;
        if (next == coordinates) {
            row_records = records;
            counters["coordinate_metadata_updates"]++;
            return;
        }
        coordinates = std::move(next);
        row_records = records;
        solver_cache.clear();
        physical_cache_valid = false;
        counters["coordinate_state_updates"]++;
    }
    std::vector<Record> combine_records(const lightstim_packed::PackedRows &coeff,
                                       const Indices &indices) const {
        if (coeff.width != indices.size()) throw std::invalid_argument("Lineage width differs from source count");
        auto records = records_selected(indices);
        std::vector<Record> out;
        for (const auto &c : coeff.data) out.push_back(combine_record_lineage(c, records));
        return out;
    }
    std::vector<Record> records() const { return row_records; }
    void replace_records(const std::vector<Record> &records) {
        if (records.size() != row_records.size()) throw std::invalid_argument("Record count differs from state row count");
        if (records != row_records) {
            relation_cache_valid = false;
            row_records = records;
            counters["record_only_reconciliations"]++;
        }
    }
    void fold_measured(const Indices &qubits, const std::vector<std::string> &bases, int64_t base_record) {
        if (qubits.size() != bases.size()) throw std::invalid_argument("Measurement qubits and bases differ in length");
        batch_active = false;
        relation_cache_valid = false;
        // Usually terminal measured Paulis already lie in the state's inverse
        // Z frame. Their factors can then be removed by coordinate XOR, without
        // exporting physical rows or re-anchoring the state. Physical X_q/Z_q
        // support is the symplectic pairing with Z_q/X_q respectively.
        struct Fold { Bits support, value; };
        std::vector<Fold> folds;
        bool coordinate_only = true;
        // In the ordinary layout each inverse image of X_q/Z_q is already a
        // contiguous packed row. Reading it directly avoids two strided full
        // inverse queries (and their dense Pauli temporaries) per ancilla.
        end();
        const size_t count_words = (n + 63) / 64;
        std::set<size_t> first_occurrence;
        std::vector<bool> unique;
        for (size_t j = 0; j < qubits.size(); j++) {
            size_t q = qubits[j];
            if (q >= n) throw std::out_of_range("Measured qubit exceeds register");
            bool ex = bases[j] == "X", ez = bases[j] == "Z";
            if (!ex && !ez) throw std::invalid_argument("Measured ancilla basis must be X or Z");
            auto measured = ex ? sim.inv_state.xs[q] : sim.inv_state.zs[q];
            auto opposite = ex ? sim.inv_state.zs[q] : sim.inv_state.xs[q];
            Bits support(count_words), value(count_words);
            for (size_t w = 0; w < count_words; w++) {
                uint64_t mask = w + 1 == count_words && n % 64
                    ? (uint64_t{1} << (n % 64)) - 1 : ~uint64_t{0};
                coordinate_only &= (measured.xs.u64[w] & mask) == 0;
                support[w] = opposite.xs.u64[w] & mask;
                value[w] = measured.zs.u64[w] & mask;
            }
            folds.push_back({std::move(support), std::move(value)});
            unique.push_back(first_occurrence.insert(q).second);
        }
        if (coordinate_only) {
            // An inverse-Z measurement commutes with every inverse-Z source.
            // Its incompatible physical component is therefore identically
            // zero. Removing a single-qubit factor cannot change support on
            // another measured qubit; each support coefficient is needed once.
            solver_cache.clear();
            physical_cache_valid = false;
            for (size_t r = 0; r < coordinates.size(); r++) {
                Record fresh;
                for (size_t j = 0; j < folds.size(); j++) {
                    if (!unique[j] || !dot(coordinates[r], folds[j].support)) continue;
                    xorin(coordinates[r], folds[j].value);
                    fresh.push_back(base_record + (int64_t)j);
                }
                // Preserve the reference's set-based normalization, including
                // literal duplicate and unknown records, but sort only once.
                auto previous = normalized(row_records[r]);
                Record combined;
                std::set_symmetric_difference(previous.begin(), previous.end(), fresh.begin(), fresh.end(),
                                              std::back_inserter(combined));
                row_records[r] = std::move(combined);
            }
            counters["folds"]++;
            counters["fold_fastpaths"]++;
            return;
        }
        auto physical = rows();
        auto records = row_records;
        for (auto &record : records) record = normalized(record);
        for (size_t j = 0; j < qubits.size(); j++) {
            size_t q = qubits[j];
            if (q >= n) throw std::out_of_range("Measured qubit exceeds register");
            bool ex = bases[j] == "X", ez = bases[j] == "Z";
            if (!ex && !ez) throw std::invalid_argument("Measured ancilla basis must be X or Z");
            for (size_t r = 0; r < coordinates.size(); r++) {
                auto *p = physical.mutable_data() + r * 2 * n;
                bool x = p[q], z = p[n + q];
                if (!x && !z) continue;
                if (x != ex || z != ez) throw std::runtime_error("Tracked row has incompatible factor on measured ancilla");
                p[q] = p[n + q] = 0;
                xor_records(records[r], Record{base_record + (int64_t)j});
            }
        }
        replace_state(physical, records);
        counters["folds"]++;
    }
    std::map<std::string, uint64_t> stats() const { return counters; }
    size_t num_qubits() const { return n; }
    py::tuple pickle_state() const {
        // rows() records an export, so serialize an independent clone to keep
        // read-only snapshots (including byte-identical repeated pickle calls)
        // from changing the live engine's instrumentation.
        PersistentRecordState snapshot(*this);
        return py::make_tuple(snapshot.rows(), row_records, counters);
    }
    static PersistentRecordState from_pickle(const py::tuple &state) {
        if (state.size() != 3) throw std::invalid_argument("Invalid PersistentRecordState pickle");
        PersistentRecordState result(state[0].cast<Rows>(), state[1].cast<std::vector<Record>>());
        result.counters = state[2].cast<std::map<std::string, uint64_t>>();
        return result;
    }
};

inline void bind_persistent_record_state(py::module_ &mod) {
    py::class_<PersistentRecordState>(mod, "PersistentRecordState")
        .def(py::init<const Rows &, const std::vector<std::vector<int64_t>> &>())
        .def("replace_state", &PersistentRecordState::replace_state)
        .def("query", &PersistentRecordState::query)
        .def("anticommuting", &PersistentRecordState::anticommuting)
        .def("decompose", &PersistentRecordState::decompose)
        .def("commuting_relation", &PersistentRecordState::commuting_relation)
        .def("begin_measurement_batch", &PersistentRecordState::begin_measurement_batch)
        .def("measurement_anticommuting", &PersistentRecordState::measurement_anticommuting)
        .def("measurement_relation", &PersistentRecordState::measurement_relation)
        .def("process_measurement_batch", &PersistentRecordState::process_measurement_batch)
        .def("project_readout_batch", &PersistentRecordState::project_readout_batch)
        .def("refresh_complement", &PersistentRecordState::refresh_complement)
        .def("refresh_sources", &PersistentRecordState::refresh_sources)
        .def("nonzero_rows", &PersistentRecordState::nonzero_rows)
        .def("ordered_complement", &PersistentRecordState::ordered_complement)
        .def("trace_qubits", &PersistentRecordState::trace_qubits)
        .def("insert_sources", &PersistentRecordState::insert_sources)
        .def("fold_rows", &PersistentRecordState::fold_rows)
        .def("adopt_physical", &PersistentRecordState::adopt_physical)
        .def("project", [](PersistentRecordState &self, const Rows &target, size_t pivot,
                           py::object record, const Indices &excluded) {
            std::vector<int64_t> records;
            if (py::isinstance<py::int_>(record)) records.push_back(record.cast<int64_t>());
            else records = record.cast<std::vector<int64_t>>();
            self.project(target, pivot, records, excluded);
        }, py::arg("target"), py::arg("pivot"),
             py::arg("record"), py::arg("excluded") = Indices{})
        .def("unitary", &PersistentRecordState::unitary)
        .def("rows", &PersistentRecordState::rows)
        .def("rows_selected", &PersistentRecordState::rows_selected)
        .def("records_selected", &PersistentRecordState::records_selected)
        .def("coordinate_rows", &PersistentRecordState::coordinate_rows)
        .def("coordinate_paulis", &PersistentRecordState::coordinate_paulis)
        .def("adopt_coordinates", &PersistentRecordState::adopt_coordinates)
        .def("combine_records", &PersistentRecordState::combine_records)
        .def("records", &PersistentRecordState::records)
        .def("replace_records", &PersistentRecordState::replace_records)
        .def("fold_measured", &PersistentRecordState::fold_measured)
        .def_property_readonly("stats", &PersistentRecordState::stats)
        .def_property_readonly("num_qubits", &PersistentRecordState::num_qubits)
        .def("copy", [](const PersistentRecordState &self) { return PersistentRecordState(self); })
        .def("__copy__", [](const PersistentRecordState &self) { return PersistentRecordState(self); })
        .def("__deepcopy__", [](const PersistentRecordState &self, py::dict) { return PersistentRecordState(self); })
        .def(py::pickle(
            [](const PersistentRecordState &self) { return self.pickle_state(); },
            [](const py::tuple &state) { return PersistentRecordState::from_pickle(state); }));
}
