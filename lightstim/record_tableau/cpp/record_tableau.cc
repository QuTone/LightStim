// Record-augmented stabilizer tableau on stim's inverse tableau (C++ engine).
//
// Engine behind lightstim/record_tableau/annotator.py.  The
// state is stim's TableauSimulator layout: inv = C^-1, so
//   * a circuit gate G is inv.prepend(G^-1) (stim's own mapping),
//   * the query q = C^-1 P C for P = Z_q / X_q is inv's row for P,
//   * an input-frame Clifford V (generator re-basing, "start of time") is
//     inv.append(V^-1); all re-basing gates are self-inverse (CX, CZ, H, H_YZ)
//     and are applied under one TableauTransposedRaii per instruction, as in
//     stim's collapse_qubit_z.
// Each generator S_k = C Z_k C^dag carries a sorted record list (its eigenvalue
// is the parity of those measurements) or UNKNOWN; logical generators carry a
// symbolic id <= LAMBDA_BASE.  Signs are never read.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "stim/gates/gates.h"
#include "stim/mem/simd_word.h"
#include "stim/simulators/tableau_simulator.h"
#include "stim/stabilizers/tableau.h"
#include "stim/stabilizers/tableau_transposed_raii.h"

namespace py = pybind11;

namespace {

constexpr size_t W = stim::MAX_BITWORD_WIDTH;
// Record values: m >= 0 is measurement m; UNMEASURED is the tracker's
// UNMEASURED_STAB_RECORD; LAMBDA_BASE - j is logical j's symbolic value
// lambda_j (the tracker's logical row j).  A value containing lambdas is the
// tracker's "logical component" (its log_vec is the set of lambdas).
constexpr int64_t LAMBDA_BASE = -(int64_t{1} << 40);
constexpr int64_t UNMEASURED = -1;

using Rec = std::vector<int64_t>; // sorted

void xor_into(Rec &dst, const Rec &src) {
  if (src.empty())
    return;
  Rec out;
  out.reserve(dst.size() + src.size());
  std::set_symmetric_difference(dst.begin(), dst.end(), src.begin(), src.end(),
                                std::back_inserter(out));
  dst.swap(out);
}

bool contains(const Rec &r, int64_t v) {
  return std::binary_search(r.begin(), r.end(), v);
}

bool has_lambda(const Rec &r) { return !r.empty() && r.front() <= LAMBDA_BASE; }

// Sorted-set helpers for generator index lists.
using Idx = std::vector<uint32_t>;

bool has(const Idx &c, uint32_t k) {
  return std::binary_search(c.begin(), c.end(), k);
}

void toggle(Idx &c, uint32_t k) {
  auto it = std::lower_bound(c.begin(), c.end(), k);
  if (it != c.end() && *it == k) {
    c.erase(it);
  } else {
    c.insert(it, k);
  }
}

Idx symdiff(const Idx &a, const Idx &b) {
  Idx out;
  std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                std::back_inserter(out));
  return out;
}

size_t intersect_count(const Idx &a, const Idx &b) {
  size_t n = 0;
  auto i = a.begin(), j = b.begin();
  while (i != a.end() && j != b.end()) {
    if (*i < *j) {
      ++i;
    } else if (*j < *i) {
      ++j;
    } else {
      ++n, ++i, ++j;
    }
  }
  return n;
}

size_t count_lambdas(const Rec &r) {
  return std::count_if(r.begin(), r.end(),
                       [](int64_t v) { return v <= LAMBDA_BASE; });
}

// dst takes on src's value: dst ^= src, or unknown when src is unknown (an
// unknown dst stays unknown).
template <typename Flag>
void absorb(Flag &dst_unknown, Rec &dst, bool src_unknown, const Rec &src) {
  if (src_unknown) {
    dst_unknown = 1;
    dst.clear();
  } else if (!dst_unknown) {
    xor_into(dst, src);
  }
}

// Dependent row: a measured check without a generator of its own (BB/toric
// redundancy), kept as a product of generators with its own records, like the
// tracker's linearly dependent stabilizer rows.
struct Dep {
  Idx c;
  Rec R;
  bool unknown = false;
};

struct Engine {
  size_t n = 0;
  // stim's simulator supplies the inverse tableau and its gate dispatch;
  // its measurement/reset paths are never used (records replace them).
  stim::TableauSimulator<W> sim{std::mt19937_64(0)};
  stim::Tableau<W> &inv = sim.inv_state;
  std::vector<Rec> R;  // generator k's value (record parity), valid unless unknown[k]
  // logical[k]: generator k is in the logical bank (tracker.logicals); all
  // other tracked generators form the stabilizer bank (tracker.stabilizers).
  std::vector<uint8_t> unknown, logical, touched;
  std::vector<int64_t> born;
  std::vector<uint32_t> batch; // generation mark of the current layer
  uint32_t batch_id = 1;
  int64_t num_measurements = 0;
  int64_t rebase_ops = 0;
  int64_t consumed_logicals =
      0; // logical generators collapsed by a measurement
  std::unique_ptr<stim::TableauTransposedRaii<W>> trans;
  std::vector<Dep> deps;
  bool allow_deps = true; // measured checks may be linearly dependent (e.g. BB)
  bool retained = false;
  Idx order;
  std::vector<std::vector<uint8_t>> clean_rows;
  std::vector<int64_t> last_record; // qubit -> record of its previous
                                    // mid-circuit measurement, or -1
  std::vector<int64_t>
      reset_qubit; // qubit whose reset created the (untouched) generator, or -1
  // Readout group (paper Alg. 2): qubit -> basis / record, deferred
  // deterministic readouts.
  std::vector<char> ro_basis;
  std::vector<int64_t> ro_rec;
  std::vector<std::tuple<uint32_t, char, int64_t>> ro_pending;
  int64_t ro_start = -1;

  explicit Engine(size_t num_qubits) { ensure_qubits(num_qubits); }

  void ensure_qubits(size_t m) {
    if (m <= n)
      return;
    if (trans)
      throw std::runtime_error("ensure_qubits inside a transposed section");
    inv.expand(m, 1.0);
    R.resize(m);
    unknown.resize(m, 1);
    logical.resize(m, 0);
    touched.resize(m, 0);
    born.resize(m, 0);
    batch.resize(m, 0);
    reset_qubit.resize(m, -1);
    last_record.resize(m, -1);
    ro_basis.resize(m, 0);
    ro_rec.resize(m, 0);
    for (size_t k = n; k < m; k++)
      order.push_back(k);
    n = m;
  }

  // ---- transposed sections (one per measurement/reset instruction) -----
  void begin() {
    trans = std::make_unique<stim::TableauTransposedRaii<W>>(inv);
  }
  void end() { trans.reset(); }

  // Column gather of inv's row for a single-qubit Pauli (valid while
  // transposed): X part = generators P anticommutes with, Z part =
  // decomposition.
  void query(char basis, size_t q, std::vector<uint32_t> &ax,
             std::vector<uint32_t> &az) const {
    ax.clear();
    az.clear();
    const auto &xs = inv.xs, &zs = inv.zs;
    for (size_t k = 0; k < n; k++) {
      bool bx, bz;
      if (basis == 'Z') {
        bx = zs.xt[k][q];
        bz = zs.zt[k][q];
      } else if (basis == 'X') {
        bx = xs.xt[k][q];
        bz = xs.zt[k][q];
      } else {
        bx = zs.xt[k][q] ^ xs.xt[k][q];
        bz = zs.zt[k][q] ^ xs.zt[k][q];
      }
      if (bx)
        ax.push_back((uint32_t)k);
      if (bz)
        az.push_back((uint32_t)k);
    }
  }

  bool touches(uint32_t k, size_t q) const {
    return inv.zs.xt[k][q] || inv.xs.xt[k][q];
  }

  // ---- values -----------------------------------------------------------
  bool value(const std::vector<uint32_t> &gens, Rec &out) const {
    out.clear();
    for (auto k : gens) {
      if (unknown[k])
        return false;
      xor_into(out, R[k]);
    }
    return true;
  }

  // ---- generator re-basing (input-frame Cliffords) ---------------------
  void mark(uint32_t g) {
    touched[g] = 1;
    reset_qubit[g] = -1;
    born[g] = num_measurements;
    batch[g] = batch_id;
  }

  void mul_into(uint32_t k, uint32_t p) { // S_k <- S_k S_p
    trans->append_ZCX(p, k);
    rebase_ops++;
    for (auto &d : deps) // S_k(old) = S_k(new) S_p
      if (has(d.c, k))
        toggle(d.c, p);
  }

  void make_generator(uint32_t g, const std::vector<uint32_t> &anti,
                      const std::vector<uint32_t> &decomp) {
    mark(g);
    if (!anti.empty()) {
      if (anti.size() != 1 || anti[0] != g)
        throw std::runtime_error("make_generator: bad pivot");
      bool y = false;
      for (auto k : decomp) {
        if (k == g) {
          y = true;
          continue;
        }
        trans->append_ZCZ(g, k);
      }
      if (y) {
        trans->append_H_YZ(g);
      } else {
        trans->append_H_XZ(g);
      }
    } else {
      for (auto k : decomp) {
        if (k != g)
          trans->append_ZCX(k, g);
      }
      for (auto &d : deps) { // S_g(old) = S_g(new) prod S_k
        if (!has(d.c, g))
          continue;
        for (auto k : decomp)
          if (k != g)
            toggle(d.c, k);
      }
    }
    rebase_ops++;
  }

  static char flip(char basis) { return basis == 'X' ? 'Z' : 'X'; }

  void strip(char basis, size_t q, uint32_t g) {
    std::vector<uint32_t> fx, fz;
    query(flip(basis), q, fx, fz);
    for (auto k : fx) {
      if (k == g)
        continue;
      absorb(unknown[k], R[k], unknown[g], R[g]);
      mul_into(k, g);
    }
  }

  // ---- policies ------------------------------------------------------------
  bool fresh(uint32_t k) const { return !unknown[k] && R[k].empty(); }

  size_t position(uint32_t k) const {
    return std::find(order.begin(), order.end(), k) - order.begin();
  }

  // Case A pivot.  The tracker takes its first anticommuting row, and a
  // logical row only when no stabilizer anticommutes; here fresh stabilizers
  // come first, logicals last.
  uint32_t pivot(const std::vector<uint32_t> &anti) const {
    uint32_t best = anti[0];
    auto key = [&](uint32_t k) {
      return std::make_tuple(logical[k], !fresh(k),
                             retained ? position(k) : size_t(k));
    };
    for (auto k : anti)
      if (key(k) < key(best))
        best = k;
    return best;
  }

  uint32_t refresh_target(const std::vector<uint32_t> &cands, size_t q) const {
    auto key = [&](uint32_t k) {
      if (unknown[k])
        return std::make_tuple(0, int64_t{0}, 0, k);
      int64_t age = R[k].empty() ? born[k] : R[k].back();
      return std::make_tuple(1, age, (int)touches(k, q), k);
    };
    uint32_t best = cands[0];
    auto bk = key(best);
    for (auto k : cands) {
      auto kk = key(k);
      if (kk < bk) {
        best = k;
        bk = kk;
      }
    }
    return best;
  }

  // Case A (process_mid_measurement): the other anticommuting generators absorb
  // the pivot, which becomes the measured Pauli with `record`.  A logical
  // pivot consumes a logical (the tracker decrements expected_num_logicals).
  uint32_t collapse(const std::vector<uint32_t> &anti, char basis, size_t q,
                    const Rec *record) {
    uint32_t p = pivot(anti);
    for (auto k : anti) {
      if (k == p)
        continue;
      absorb(unknown[k], R[k], unknown[p], R[p]);
      mul_into(k, p);
    }
    for (auto &d : deps) { // anticommuting rows absorb the pivot
      if (!has(d.c, p))
        continue;
      toggle(d.c, p);
      absorb(d.unknown, d.R, unknown[p], R[p]);
    }
    if (logical[p])
      consumed_logicals++;
    logical[p] = 0;
    std::vector<uint32_t> a2, d2;
    query(basis, q, a2, d2);
    make_generator(p, a2, d2);
    unknown[p] = 0;
    R[p] = record ? *record : Rec{};
    return p;
  }

  // A deterministic outcome involving an unknown generator teaches its value:
  // that generator becomes the measured Pauli, holding record m.
  bool learn_unknown(const Idx &decomp, int64_t m, char basis, size_t q,
                     bool do_strip) {
    auto u = std::find_if(decomp.begin(), decomp.end(),
                          [&](uint32_t k) { return unknown[k]; });
    if (u == decomp.end())
      return false;
    uint32_t g = *u;
    make_generator(g, {}, decomp);
    unknown[g] = 0;
    R[g] = Rec{m};
    if (do_strip)
      strip(basis, q, g);
    return true;
  }

  // Re-base so S_g is the measured Pauli holding outcome m (write-back), and
  // remove that factor from the other generators.
  void anchor_measurement(uint32_t g, const Idx &decomp, int64_t m, char basis,
                          size_t q) {
    logical[g] = 0;
    make_generator(g, {}, decomp);
    unknown[g] = 0;
    R[g] = Rec{m};
    strip(basis, q, g);
  }

  // ---- instructions ------------------------------------------------------
  void reset(char basis, const std::vector<uint32_t> &targets) {
    begin();
    std::vector<uint32_t> anti, decomp, fx, fz;
    for (auto q : targets) {
      query(basis, q, anti, decomp);
      uint32_t g;
      bool v_known;
      Rec v;
      if (!anti.empty()) {
        g = collapse(anti, basis, q, nullptr);
        v_known = false; // hidden random outcome
      } else {
        v_known = value(decomp, v);
        g = retained ? *std::min_element(decomp.begin(), decomp.end(),
                                         [&](uint32_t a, uint32_t b) {
                                           return position(a) < position(b);
                                         })
                     : refresh_target(decomp, q);
        make_generator(g, {}, decomp);
      }
      if (!v_known) {
        order.erase(std::find(order.begin(), order.end(), g));
        order.push_back(g);
      }
      query(flip(basis), q, fx, fz);
      for (auto k : fx)
        if (k != g)
          absorb(unknown[k], R[k], !v_known, v);
      for (auto &d : deps)
        if (intersect_count(d.c, fx) % 2)
          absorb(d.unknown, d.R, !v_known, v);
      unknown[g] = 0;
      R[g].clear();
      if (retained)
        strip(basis, q, g);
      touched[g] = 0; // a reset creates a fresh constraint
      reset_qubit[g] = q;
    }
    end();
  }

  using Relations = std::vector<std::pair<int64_t, Rec>>;

  // Mid-circuit measurement layer (one basis, or one basis per target):
  // detectors against the layer input; re-anchoring deferred to the layer end.
  Relations measure(const std::string &bases,
                    const std::vector<uint32_t> &targets) {
    if (bases.size() != 1 && bases.size() != targets.size())
      throw std::invalid_argument("measure: need one basis or one per target");
    begin();
    batch_id++;
    int64_t start = num_measurements;
    if (retained) {
      std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return std::make_tuple(logical[a], !fresh(a)) <
               std::make_tuple(logical[b], !fresh(b));
      });
    }
    Relations out;
    std::vector<std::tuple<char, size_t, int64_t, int>> pending;
    std::vector<uint32_t> anti, decomp;
    Rec val;
    for (size_t i = 0; i < targets.size(); i++) {
      size_t q = targets[i];
      char basis = bases.size() == 1 ? bases[0] : bases[i];
      if (basis != 'X' && basis != 'Y' && basis != 'Z')
        throw std::invalid_argument("basis must be X, Y or Z");
      int64_t m = num_measurements++;
      query(basis, q, anti, decomp);
      if (!anti.empty()) { // Case A: anticommutes, state update
        Rec rec{m};
        uint32_t p = collapse(anti, basis, q, &rec);
        if (!retained)
          strip(basis, q, p);
        continue;
      }
      if (learn_unknown(decomp, m, basis, q, !retained))
        continue;
      value(decomp, val); // Case B: commutes; its decomposition is decomp, no RREF
      xor_into(val, Rec{m});
      int dep = matching_dep(decomp);
      if (dep >= 0) { // tracker's exact-match row
        Rec alt;
        value(symdiff(decomp, deps[dep].c), alt);
        xor_into(alt, deps[dep].R);
        xor_into(alt, Rec{m});
        val = alt;
      }
      out.emplace_back(m, val);
      pending.emplace_back(basis, q, m, dep);
    }
    if (retained) {
      Idx claimed;
      std::vector<uint8_t> used(n, 0);
      for (size_t i = 0; i < targets.size(); i++) {
        char basis = bases.size() == 1 ? bases[0] : bases[i];
        if (!clean_rows.empty()) {
          const auto &row = clean_rows.at(i);
          query_row(row.data(), row.size() / 2, anti, decomp);
        } else {
          query(basis, targets[i], anti, decomp);
        }
        uint32_t g = UINT32_MAX;
        for (auto k : decomp) {
          if (used[k])
            continue;
          if (g == UINT32_MAX || std::make_tuple(!logical[k], position(k)) >
                                     std::make_tuple(!logical[g], position(g)))
            g = k;
        }
        if (g == UINT32_MAX)
          continue;
        make_generator(g, {}, decomp);
        R[g] = Rec{start + (int64_t)i};
        unknown[g] = logical[g] = 0;
        used[g] = 1;
        claimed.push_back(g);
      }
      for (auto k : order)
        if (!used[k])
          claimed.push_back(k);
      order.swap(claimed);
      for (size_t i = 0; i < targets.size(); i++)
        last_record[targets[i]] = start + (int64_t)i;
      end();
      return out;
    }
    // Deferred write-back, two passes: every check first re-anchors the
    // generator holding its own previous outcome (so a Bell partner cannot
    // take it), then the rest use the oldest-record rule.
    std::vector<size_t> rest;
    for (size_t i = 0; i < pending.size(); i++)
      if (!refresh_one(pending[i], start, true))
        rest.push_back(i);
    for (auto i : rest)
      refresh_one(pending[i], start, false);
    for (size_t i = 0; i < targets.size(); i++)
      last_record[targets[i]] = start + (int64_t)i;
    end();
    return out;
  }

  bool refresh_one(const std::tuple<char, size_t, int64_t, int> &item,
                   int64_t start, bool own_only) {
    auto [basis, q, m, dep] = item;
    Idx anti, decomp, freshl, cands, mine;
    query(basis, q, anti, decomp);
    if (!anti.empty())
      throw std::runtime_error(
          "deferred refresh: measurement stopped commuting");
    for (auto k : decomp) {
      if (batch[k] == batch_id)
        continue;
      if (!unknown[k] && !R[k].empty() && R[k].back() >= start)
        continue;
      freshl.push_back(k);
    }
    for (auto k : freshl)
      if (!logical[k])
        cands.push_back(k);
    if (cands.empty())
      cands = freshl;
    int64_t own = last_record[q];
    // The generator holding this qubit's previous outcome stays eligible even
    // if a Bell partner's strip in this layer added a current record to it.
    for (auto k : decomp)
      if (batch[k] != batch_id && !logical[k] && own >= 0 && !unknown[k] &&
          contains(R[k], own))
        mine.push_back(k);
    if (own_only && mine.empty())
      return false;
    if (dep >= 0) {
      Rec r;
      value(symdiff(decomp, deps[dep].c), r);
      xor_into(r, Rec{m});
      deps[dep].R = r;
    }
    if (mine.empty() && cands.empty())
      return true;
    Rec joint_value;
    if (value(decomp, joint_value) && count_lambdas(joint_value) > 1) {
      // A joint outcome (two or more lambdas: the tracker's absorbed logical
      // relation) consumes a logical direction, which becomes a stabilizer
      // anchored to the current measurement.
      Idx joint;
      for (auto k : decomp)
        if (logical[k])
          joint.push_back(k);
      if (!joint.empty()) {
        mine.clear();
        cands.swap(joint);
      }
    }
    uint32_t g = refresh_target(mine.empty() ? cands : mine,
                                q); // each check keeps its own anchor
    if (allow_deps && dep < 0 && freshl.size() == 1 && freshl[0] == g &&
        reset_qubit[g] == (int64_t)q && decomp.size() > 1 && !unknown[g]) {
      // Spanned by this layer's other checks: keep the data-only form
      // P * S_g as a dependent row.
      Dep d;
      for (auto k : decomp)
        if (k != g)
          d.c.push_back(k);
      d.R = R[g];
      xor_into(d.R, Rec{m});
      deps.push_back(std::move(d));
    }
    anchor_measurement(g, decomp, m, basis, q);
    return true;
  }

  int matching_dep(const std::vector<uint32_t> &decomp) const {
    int best = -1;
    size_t best_w = 0;
    for (size_t i = 0; i < deps.size(); i++) {
      if (deps[i].unknown)
        continue;
      size_t w = decomp.size() + deps[i].c.size() -
                 2 * intersect_count(decomp, deps[i].c);
      if (w < decomp.size() && (best < 0 || w < best_w)) {
        best = (int)i;
        best_w = w;
      }
    }
    return best;
  }

  // Data readout (paper Alg. 2), Rule-2 part: sequential Case-A updates;
  // deterministic readouts are deferred to finish_readout, which closes the
  // readout group (one or more measurement layers).
  Relations readout(char basis, const std::vector<uint32_t> &targets) {
    begin();
    std::vector<uint32_t> anti, decomp;
    for (auto q : targets) {
      int64_t m = num_measurements++;
      if (ro_start < 0)
        ro_start = m;
      ro_basis[q] = basis;
      ro_rec[q] = m;
      query(basis, q, anti, decomp);
      if (!anti.empty()) {
        Rec rec{m};
        uint32_t p = collapse(anti, basis, q, &rec);
        strip(basis, q, p);
        continue;
      }
      if (learn_unknown(decomp, m, basis, q, true))
        continue;
      ro_pending.emplace_back(q, basis, m);
    }
    end();
    return {};
  }

  // Pass 1 / 2 relation of a physical Pauli carrying records r:
  // valid iff its support lies on the group's readouts in the readout bases.
  bool pass1(stim::PauliStringRef<W> pauli, const Rec &r, Rec &val) const {
    val = r;
    bool valid = true;
    pauli.for_each_active_pauli([&](size_t q) {
      if (!valid)
        return;
      if (q >= n || !ro_basis[q] || pauli.xs[q] != (ro_basis[q] != 'Z') ||
          pauli.zs[q] != (ro_basis[q] != 'X')) {
        valid = false;
      } else {
        xor_into(val, Rec{ro_rec[q]});
      }
    });
    return valid;
  }

  // Close the readout group: Pass-1 relations first, completed by the pending
  // readouts' own relations up to one independent parity per deterministic
  // readout; then re-anchor the pending readouts.
  Relations finish_readout() {
    Relations out;
    if (ro_start < 0)
      return out;
    int64_t lo = ro_start;
    auto fwd = inv.inverse(true); // forward tableau: S_k = fwd.zs[k]
    std::vector<Rec> cands;
    Rec val;
    for (size_t k = 0; k < n; k++) {
      if (unknown[k] || contains(R[k], UNMEASURED))
        continue; // tracker skips unmeasured rows
      if (pass1(fwd.zs[k], R[k], val) && !val.empty())
        cands.push_back(val);
    }
    // Dependent rows (the tracker's redundant rows) are emitted as they are.
    std::vector<Dep> kept;
    stim::PauliString<W> product(n);
    for (auto &d : deps) {
      if (d.unknown || contains(d.R, UNMEASURED)) {
        kept.push_back(d);
        continue;
      }
      product.xs.clear();
      product.zs.clear();
      for (auto k : d.c) {
        product.xs ^= fwd.zs[k].xs;
        product.zs ^= fwd.zs[k].zs;
      }
      if (!pass1(product, d.R, val)) {
        kept.push_back(d);
      } else if (!val.empty()) {
        out.emplace_back(val.back(), val);
      }
    }
    deps.swap(kept);
    size_t n_dep = out.size();

    begin();
    std::vector<uint32_t> anti, decomp;
    std::vector<std::pair<bool, Rec>> seq;
    for (auto &[q, basis, m] : ro_pending) {
      query(basis, q, anti, decomp);
      bool unmeasured = false;
      for (auto k : decomp)
        unmeasured |= contains(R[k], UNMEASURED);
      if (unmeasured) {
        seq.emplace_back(false, Rec{});
        continue;
      }
      value(decomp, val);
      xor_into(val, Rec{m});
      seq.emplace_back(true, val);
      cands.push_back(val);
    }
    std::map<int64_t, Rec> rows;
    for (auto &c : cands) {
      if (out.size() - n_dep == ro_pending.size())
        break;
      Rec v(std::lower_bound(c.begin(), c.end(), lo), c.end());
      while (!v.empty()) {
        auto it = rows.find(v.back());
        if (it == rows.end())
          break;
        xor_into(v, it->second);
      }
      if (!v.empty()) {
        rows[v.back()] = v;
        out.emplace_back(c.back(), c);
      }
    }
    batch_id++;
    std::vector<uint32_t> cands_g;
    for (size_t i = 0; i < ro_pending.size(); i++) { // deferred write-back
      auto &[q, basis, m] = ro_pending[i];
      query(basis, q, anti, decomp);
      uint32_t first_logical = UINT32_MAX;
      for (auto k : decomp)
        if (logical[k]) {
          first_logical = k;
          break;
        }
      uint32_t g;
      if (seq[i].first && has_lambda(seq[i].second) &&
          first_logical != UINT32_MAX) {
        g = first_logical;
      } else {
        cands_g.clear();
        for (auto k : decomp)
          if (!logical[k])
            cands_g.push_back(k);
        if (cands_g.empty())
          cands_g = decomp;
        g = refresh_target(cands_g, q);
      }
      anchor_measurement(g, decomp, m, basis, q);
    }
    end();
    std::fill(ro_basis.begin(), ro_basis.end(), 0);
    ro_pending.clear();
    ro_start = -1;
    return out;
  }

  // Gates: stim's TableauSimulator::do_gate on the shared inverse tableau
  // (inv.prepend(G^-1)); stim resolves aliases and inverses.
  void unitary(const std::string &name, const std::vector<uint32_t> &t) {
    if (trans)
      throw std::runtime_error("unitary inside a transposed section");
    const stim::Gate *gate;
    try {
      gate = &stim::GATE_DATA.at(name);
    } catch (const std::out_of_range &) {
      throw std::invalid_argument("record tableau: unknown gate " + name);
    }
    if (!(gate->flags & stim::GATE_IS_UNITARY) ||
        (gate->flags & stim::GATE_TARGETS_PAULI_STRING))
      throw std::invalid_argument("record tableau: unsupported gate " + name);
    std::vector<stim::GateTarget> targets;
    targets.reserve(t.size());
    for (auto q : t)
      targets.push_back(stim::GateTarget::qubit(q));
    sim.do_gate(stim::CircuitInstruction(gate->id, {}, targets, ""));
  }

  // ---- logical classification -------------------------------------------
  // Initialised constraints no measurement has re-anchored since: the rows
  // process_mid_measurement reports as promotable to logicals.
  std::vector<uint32_t> promotable_stabilizers() const {
    std::vector<uint32_t> out;
    for (size_t k = 0; k < n; k++)
      if (!touched[k] && !logical[k] && fresh((uint32_t)k))
        out.push_back((uint32_t)k);
    return out;
  }

  // tracker.promote_stabilizer_rows_to_logicals: move them to the logical
  // bank, with symbols lambda_first_lambda, lambda_first_lambda+1, ...
  void promote_stabilizers_to_logicals(const std::vector<uint32_t> &gens,
                                       int64_t first_lambda) {
    for (size_t j = 0; j < gens.size(); j++) {
      auto g = gens[j];
      logical[g] = 1;
      unknown[g] = 0;
      R[g] = Rec{LAMBDA_BASE - (first_lambda + (int64_t)j)};
    }
  }

  // (anticommuting generators, decomposition) of a Pauli given as an [X|Z]
  // bit row of `width` qubits: XOR of its single-qubit queries (transposed).
  void query_row(const uint8_t *row, size_t width, Idx &anti, Idx &decomp) {
    std::vector<uint8_t> acc_x(n, 0), acc_z(n, 0);
    Idx qa, qz;
    for (size_t q = 0; q < width; q++) {
      bool x = row[q], z = row[width + q];
      if (!x && !z)
        continue;
      query(x && z ? 'Y' : (x ? 'X' : 'Z'), q, qa, qz);
      for (auto k : qa)
        acc_x[k] ^= 1;
      for (auto k : qz)
        acc_z[k] ^= 1;
    }
    anti.clear();
    decomp.clear();
    for (size_t k = 0; k < n; k++) {
      if (acc_x[k])
        anti.push_back((uint32_t)k);
      if (acc_z[k])
        decomp.push_back((uint32_t)k);
    }
  }

  using Rows = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

  // [X|Z] bit rows passed from numpy: one row (ndim 1) or a 2-D array.
  struct RowArray {
    const uint8_t *data;
    int ndim;
    size_t count, width, cols;
    const uint8_t *operator[](size_t i) const { return data + i * 2 * width; }
  };

  // Views the rows and widens the engine to cover them.
  RowArray row_array(const Rows &rows) {
    auto buf = rows.request();
    RowArray a{static_cast<const uint8_t *>(buf.ptr), (int)buf.ndim, 0, 0, 0};
    if (buf.ndim >= 1)
      a.cols = buf.shape[buf.ndim - 1];
    if (buf.ndim == 1) {
      a.count = 1;
      a.width = a.cols / 2;
    } else if (buf.ndim == 2) {
      a.count = buf.shape[0];
      a.width = a.count ? a.cols / 2 : 0;
    }
    ensure_qubits(a.width);
    return a;
  }

  RowArray single_row(const Rows &row) {
    RowArray a = row_array(row);
    if (a.ndim != 1 || a.cols % 2)
      throw std::invalid_argument("Expected a single symplectic row.");
    return a;
  }

  void anchor_checks(Rows rows, const std::vector<Rec> &records) {
    RowArray a = row_array(rows);
    if (a.ndim != 2 || a.count != records.size())
      throw std::invalid_argument(
          "Expected one record value per output check.");
    begin();
    std::vector<uint8_t> claimed(n, 0);
    Idx anti, decomp;
    Rec old;
    for (size_t r = 0; r < records.size(); r++) {
      query_row(a[r], a.width, anti, decomp);
      if (!anti.empty() || !value(decomp, old) || has_lambda(old))
        continue;
      uint32_t g = UINT32_MAX;
      for (auto k : decomp) {
        if (claimed[k])
          continue;
        if (g == UINT32_MAX || (logical[g] && !logical[k]))
          g = k;
      }
      if (g == UINT32_MAX)
        continue;
      make_generator(g, {}, decomp);
      R[g] = records[r];
      unknown[g] = logical[g] = 0;
      claimed[g] = 1;
    }
    end();
  }

  py::object logical_value(Rows row) {
    RowArray a = single_row(row);
    begin();
    Idx anti, decomp;
    Rec v;
    query_row(a[0], a.width, anti, decomp);
    bool known = anti.empty() && value(decomp, v);
    end();
    if (!known || count_lambdas(v) != 1)
      return py::none();
    return py::frozenset(py::cast(v));
  }

  void anchor_logical(Rows row, const Rec &v) {
    RowArray a = single_row(row);
    begin();
    Idx anti, decomp;
    query_row(a[0], a.width, anti, decomp);
    auto g = std::find_if(decomp.begin(), decomp.end(),
                          [&](uint32_t k) { return logical[k]; });
    if (!anti.empty() || g == decomp.end()) {
      end();
      throw std::invalid_argument(
          "Logical representative is not in the tracked frame.");
    }
    make_generator(*g, {}, decomp);
    R[*g] = v;
    unknown[*g] = 0;
    end();
  }

  void shift_logical_records(int64_t symbol, const Rec &delta) {
    for (size_t k = 0; k < n; k++)
      if (!unknown[k] && contains(R[k], symbol))
        xor_into(R[k], delta);
    for (auto &d : deps)
      if (!d.unknown && contains(d.R, symbol))
        xor_into(d.R, delta);
  }

  // builder._try_canonicalize_stateful_code_frame: if every code stabilizer is
  // in the tracked stabilizer group, re-anchor them -- and the single-qubit
  // constraints `aux_rows` on non-data qubits that the state satisfies -- as
  // generators, then label `need` declared logicals that are in the group and
  // independent of them (they keep their records).  Returns the number
  // labelled (0: unchanged).
  int canonicalize_stateful_code_frame(Rows stab_rows, Rows logical_rows,
                                       int need, int64_t first_lambda,
                                       py::object aux_obj) {
    if (need <= 0)
      return 0;
    RowArray L = row_array(logical_rows);
    if (L.ndim != 2 || L.count == 0)
      return 0;
    RowArray S = row_array(stab_rows);
    Rows aux_rows = aux_obj.is_none() ? Rows(std::vector<py::ssize_t>{0, 0})
                                      : aux_obj.cast<Rows>();
    RowArray A = row_array(aux_rows);
    size_t ns = S.ndim == 2 ? S.count : 0, na = A.ndim == 2 ? A.count : 0;
    begin();
    Idx anti, decomp;
    auto tracked = [&](const Idx &a, const Idx &d) {
      if (!a.empty())
        return false;
      for (auto k : d)
        if (unknown[k] || logical[k])
          return false;
      return true;
    };
    std::map<uint32_t, Idx> basis;
    auto reduce = [&](Idx v) {
      while (!v.empty()) {
        auto it = basis.find(v.back());
        if (it == basis.end())
          break;
        v = symdiff(v, it->second);
      }
      return v;
    };
    for (size_t r = 0; r < ns; r++) {
      query_row(S[r], S.width, anti, decomp);
      if (!tracked(anti, decomp)) {
        end();
        return 0;
      }
      Idx v = reduce(decomp);
      if (!v.empty())
        basis[v.back()] = v;
    }
    std::vector<size_t> aux;
    for (size_t r = 0; r < na; r++) {
      query_row(A[r], A.width, anti, decomp);
      if (!tracked(anti, decomp))
        continue;
      Idx v = reduce(decomp);
      if (!v.empty()) {
        basis[v.back()] = v;
        aux.push_back(r);
      }
    }
    std::vector<size_t> accepted;
    for (size_t r = 0; r < L.count; r++) {
      query_row(L[r], L.width, anti, decomp);
      if (!tracked(anti, decomp))
        continue;
      Idx v = reduce(decomp);
      if (!v.empty()) {
        basis[v.back()] = v;
        accepted.push_back(r);
      }
    }
    if ((int)accepted.size() != need) {
      end();
      return 0;
    }
    std::vector<uint8_t> claimed(n, 0);
    Rec v;
    for (size_t i = 0; i < ns + aux.size(); i++) {
      if (i < ns) {
        query_row(S[i], S.width, anti, decomp);
      } else {
        query_row(A[aux[i - ns]], A.width, anti, decomp);
      }
      uint32_t g = UINT32_MAX;
      for (auto k : decomp)
        if (!claimed[k]) {
          g = k;
          break;
        }
      if (g == UINT32_MAX)
        continue;
      value(decomp, v);
      make_generator(g, {}, decomp);
      unknown[g] = 0;
      R[g] = v;
      claimed[g] = 1;
    }
    for (size_t j = 0; j < accepted.size(); j++) {
      query_row(L[accepted[j]], L.width, anti, decomp);
      uint32_t g = UINT32_MAX;
      for (auto k : decomp)
        if (!claimed[k]) {
          g = k;
          break;
        }
      value(decomp, v);
      make_generator(g, {}, decomp);
      Rec lam{LAMBDA_BASE - (first_lambda + (int64_t)j)};
      xor_into(v, lam);
      unknown[g] = 0;
      R[g] = v;
      logical[g] = 1;
      claimed[g] = 1;
    }
    end();
    return need;
  }

  // tracker.rebase_stabilizers_onto_code_basis (and, with tag_unmeasured, the
  // UNMEASURED rows of stabilizer_canonicalization): re-anchor one fresh
  // generator to each code stabilizer the state satisfies.
  void rebase_onto_code_basis(Rows rows, bool tag_unmeasured) {
    RowArray a = row_array(rows);
    if (a.ndim != 2 || a.count == 0)
      return;
    begin();
    std::vector<uint32_t> anti, decomp, fr;
    Rec v;
    for (size_t r = 0; r < a.count; r++) {
      query_row(a[r], a.width, anti, decomp);
      if (!anti.empty())
        continue;
      fr.clear();
      for (auto k : decomp)
        if (!touched[k] && !logical[k] && fresh(k))
          fr.push_back(k);
      if (fr.empty())
        continue;
      bool known = value(decomp, v);
      uint32_t g = fr[0];
      make_generator(g, {}, decomp);
      if (known) {
        if (tag_unmeasured && !contains(v, UNMEASURED)) {
          v.insert(std::lower_bound(v.begin(), v.end(), UNMEASURED),
                   UNMEASURED);
        }
        unknown[g] = 0;
        R[g] = v;
      } else {
        unknown[g] = 1;
        R[g].clear();
      }
    }
    end();
  }
};

py::list relations_to_py(const Engine::Relations &rel) {
  py::list out;
  for (auto &[m, r] : rel)
    out.append(py::make_tuple(m, py::frozenset(py::cast(r))));
  return out;
}

char basis_char(const std::string &b) {
  if (b == "X" || b == "Y" || b == "Z")
    return b[0];
  throw std::invalid_argument("basis must be X, Y or Z");
}

} // namespace

PYBIND11_MODULE(_record_tableau_cpp, m) {
  m.doc() = "Record-augmented stabilizer tableau on stim's inverse tableau.";
  m.attr("SIMD_WIDTH") = (int)W;
  py::class_<Engine>(m, "Engine")
      .def(py::init<size_t>(), py::arg("num_qubits") = 0)
      .def("ensure_qubits", &Engine::ensure_qubits)
      .def("unitary", &Engine::unitary)
      .def("reset",
           [](Engine &e, const std::string &b, const std::vector<uint32_t> &t) {
             e.reset(basis_char(b), t);
           })
      .def("measure",
           [](Engine &e, const std::string &b, const std::vector<uint32_t> &t) {
             return relations_to_py(e.measure(b, t));
           })
      .def("readout",
           [](Engine &e, const std::string &b, const std::vector<uint32_t> &t) {
             return relations_to_py(e.readout(basis_char(b), t));
           })
      .def("finish_readout",
           [](Engine &e) { return relations_to_py(e.finish_readout()); })
      .def("promotable_stabilizers", &Engine::promotable_stabilizers)
      .def("anchor_checks", &Engine::anchor_checks)
      .def("logical_value", &Engine::logical_value)
      .def("anchor_logical", &Engine::anchor_logical)
      .def("shift_logical_records", &Engine::shift_logical_records)
      .def("promote_stabilizers_to_logicals",
           &Engine::promote_stabilizers_to_logicals)
      .def("rebase_onto_code_basis", &Engine::rebase_onto_code_basis,
           py::arg("rows"), py::arg("tag_unmeasured") = false)
      .def("canonicalize_stateful_code_frame",
           &Engine::canonicalize_stateful_code_frame, py::arg("stab_rows"),
           py::arg("logical_rows"), py::arg("need"), py::arg("first_lambda"),
           py::arg("aux_rows") = py::none())
      .def("snapshot",
           [](const Engine &e) {
             // Read-only export of the tracked state for tracker_view.py: [X|Z]
             // rows of the tracked generators, their indices, records (incl.
             // lambda ids), logical flags, dependent rows (generators, records
             // or None) and the number of consumed logicals.
             if (e.trans)
               throw std::runtime_error("snapshot inside a transposed section");
             auto fwd = e.inv.inverse(true); // S_k = fwd.zs[k]
             std::vector<uint32_t> gens;
             for (size_t k = 0; k < e.n; k++)
               if (!e.unknown[k])
                 gens.push_back(k);
             py::array_t<uint8_t> rows(
                 {(py::ssize_t)gens.size(), (py::ssize_t)(2 * e.n)});
             auto r = rows.mutable_unchecked<2>();
             py::list recs, kinds, deps;
             for (size_t i = 0; i < gens.size(); i++) {
               auto z = fwd.zs[gens[i]];
               for (size_t q = 0; q < e.n; q++) {
                 r(i, q) = (bool)z.xs[q];
                 r(i, e.n + q) = (bool)z.zs[q];
               }
               recs.append(py::cast(e.R[gens[i]]));
               kinds.append((bool)e.logical[gens[i]]);
             }
             for (const auto &d : e.deps)
               deps.append(py::make_tuple(py::cast(d.c),
                                          d.unknown ? py::object(py::none())
                                                    : py::cast(d.R)));
             return py::make_tuple(rows, py::cast(gens), recs, kinds, deps,
                                   e.consumed_logicals);
           })
      .def_readwrite("allow_deps", &Engine::allow_deps)
      .def_readwrite("retained", &Engine::retained)
      .def_readwrite("clean_rows", &Engine::clean_rows)
      .def_readonly("n", &Engine::n)
      .def_readonly("num_measurements", &Engine::num_measurements)
      .def_readonly("rebase_ops", &Engine::rebase_ops);
}
