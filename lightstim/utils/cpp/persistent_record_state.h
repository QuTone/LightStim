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
    std::map<std::string, uint64_t> counters;
    stim::TableauTransposedRaii<W> &begin() {
        if (!transposed) transposed = std::make_unique<stim::TableauTransposedRaii<W>>(sim.inv_state);
        return *transposed;
    }
    void end() const { transposed.reset(); }

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
        for (size_t k = 0; k < n; k++) {
            bool ax = false, bz = false;
            for (size_t q : x) { ax ^= inv.xs.xt[k][q]; bz ^= inv.xs.zt[k][q]; }
            for (size_t q : z) { ax ^= inv.zs.xt[k][q]; bz ^= inv.zs.zt[k][q]; }
            if (ax) flip(a, k);
            if (bz) flip(b, k);
        }
    }
    void anchor(stim::TableauTransposedRaii<W> &trans, size_t g,
                const Bits &a, const Bits &b) {
        if (bit(a, g)) {
            for (size_t k = 0; k < n; k++)
                if (k != g && bit(b, k)) { trans.append_ZCZ(g, k); counters["frame_rebases"]++; }
            if (bit(b, g)) trans.append_H_YZ(g);
            else trans.append_H_XZ(g);
            counters["frame_rebases"]++;
        } else {
            for (size_t k = 0; k < n; k++)
                if (k != g && bit(b, k)) { trans.append_ZCX(k, g); counters["frame_rebases"]++; }
        }
    }
    // Change S_dst to S_dst*S_src and update coordinates of every retained row.
    void multiply_generator(stim::TableauTransposedRaii<W> &trans, size_t src, size_t dst) {
        trans.append_ZCX(src, dst);
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
    py::tuple decompose_coordinates(const Bits &a, Bits b, const Indices &indices) const {
        py::array_t<uint8_t> coeff(coordinates.size());
        std::fill_n(coeff.mutable_data(), coordinates.size(), uint8_t{0});
        for (uint64_t w : a) if (w) return py::make_tuple(coeff, false, Record{});
        auto entry = solver_cache.find(indices);
        if (entry == solver_cache.end()) {
            if (solver_cache.size() >= 8) solver_cache.clear();
            entry = solver_cache.emplace(indices, make_solver(indices)).first;
        }
        const auto &solver = entry->second;
        Bits selected((coordinates.size() + 63) / 64, 0);
        for (size_t p = 0; p < n; p++) {
            if (!bit(b, p)) continue;
            if (solver.pivots[p].empty()) return py::make_tuple(coeff, false, Record{});
            xorin(b, solver.pivots[p]); xorin(selected, solver.coeffs[p]);
        }
        Record occurrences;
        bool unknown = false;
        for (size_t k = 0; k < coordinates.size(); k++) if (bit(selected, k)) {
            coeff.mutable_data()[k] = 1;
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
        return py::make_tuple(coeff, true, record);
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
                    trans.append_ZCX(g, k); counters["frame_rebases"]++;
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
    void project(const Rows &target, size_t pivot, const Record &record, const Indices &excluded = {}) {
        const auto *row = validate_target(target);
        if (pivot >= coordinates.size()) throw std::out_of_range("Projection pivot out of range");
        counters["projections"]++;
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
        end();
        counters["snapshots"]++;
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
        py::array_t<uint8_t> result({(py::ssize_t)coordinates.size(), (py::ssize_t)(2 * n)});
        for (size_t r = 0; r < coordinates.size(); r++) {
            Bits physical(words, 0);
            for (size_t k = 0; k < n; k++) if (bit(coordinates[r], k)) xorin(physical, packed[k]);
            for (size_t k = 0; k < 2 * n; k++) result.mutable_data()[r * 2 * n + k] = bit(physical, k);
        }
        return result;
    }
    std::vector<Record> records() const { return row_records; }
    void fold_measured(const Indices &qubits, const std::vector<std::string> &bases, int64_t base_record) {
        if (qubits.size() != bases.size()) throw std::invalid_argument("Measurement qubits and bases differ in length");
        // Usually terminal measured Paulis already lie in the state's inverse
        // Z frame. Their factors can then be removed by coordinate XOR, without
        // exporting physical rows or re-anchoring the state. Physical X_q/Z_q
        // support is the symplectic pairing with Z_q/X_q respectively.
        struct Fold { Bits x_support, z_support, value; bool x, z; };
        std::vector<Fold> folds;
        bool coordinate_only = true;
        begin();
        for (size_t j = 0; j < qubits.size(); j++) {
            size_t q = qubits[j];
            if (q >= n) throw std::out_of_range("Measured qubit exceeds register");
            bool ex = bases[j] == "X", ez = bases[j] == "Z";
            if (!ex && !ez) throw std::invalid_argument("Measured ancilla basis must be X or Z");
            std::vector<uint8_t> pauli(2 * n, 0);
            Bits ax, bx, az, bz;
            pauli[q] = 1; query_coordinates(pauli.data(), ax, bx);
            pauli[q] = 0; pauli[n + q] = 1; query_coordinates(pauli.data(), az, bz);
            const auto &a = ex ? ax : az;
            for (uint64_t w : a) coordinate_only &= w == 0;
            folds.push_back({az, ax, ex ? bx : bz, ex, ez});
        }
        if (coordinate_only) {
            for (const auto &c : coordinates) for (const auto &f : folds) {
                bool x = dot(c, f.x_support), z = dot(c, f.z_support);
                if ((x || z) && (x != f.x || z != f.z))
                    throw std::runtime_error("Tracked row has incompatible factor on measured ancilla");
            }
            solver_cache.clear();
            for (auto &record : row_records) record = normalized(record);
            for (size_t r = 0; r < coordinates.size(); r++) for (size_t j = 0; j < folds.size(); j++) {
                const auto &f = folds[j];
                if (!dot(coordinates[r], f.x_support) && !dot(coordinates[r], f.z_support)) continue;
                xorin(coordinates[r], f.value);
                xor_records(row_records[r], Record{base_record + (int64_t)j});
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
        .def("records", &PersistentRecordState::records)
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
