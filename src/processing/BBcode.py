from typing import Tuple, Set, Dict, List
import stim

def transpose_exponents(monomials, l, m):
    return [[(-i) % l, (-j) % m] for i, j in monomials]



def multiply_monomial_by_polynomial(A, B, l, m,type = 'L'):
    i, j = A
    if type == 'L':
        return [(2 *((i + p) % l), 2* ((j + q) % m) + 1) for p, q in B]
    elif type == 'R':
        return [(2* ((i + p) % l) + 1, 2* ((j + q) % m)) for p, q in B]
    else:
        raise ValueError("Type must be either 'L' or 'R'.")
 

class BBcode:
    def __init__(self, n: int, k: int, d: int, m: int, l: int, A: list[list[int]], B: list[list[int]], shift: list[int],f: list[list[int]], g: list[list[int]], h: list[list[int]], alpha:list[list[int]], beta: list[list[int]]):
        self.n = n
        self.k = k
        self.d = d
        self.m = m
        self.l = l
        self.A = A
        self.B = B
        self.shift = shift
        self.f = f
        self.g = g
        self.h = h
        self.alpha = alpha
        self.beta = beta

        self.data_coords = set()
        self.ancilla_coords_x = set()
        self.ancilla_coords_z = set()
        self.stabilizers = []
        self.logical_X = []
        self.logical_Z = []

        self._generate_BBcode()
        if shift != [0,0]:
            self.patch_shift(shift)

    def _generate_BBcode(self):
        n = self.n
        k = self.k
        d = self.d
        m = self.m
        l = self.l
        A = self.A
        B = self.B
        f = self.f
        g = self.g
        h = self.h
        alpha = self.alpha
        beta = self.beta

        #construct the BB code with ancilla and data qubits for 2d layout
        for y in range(2*m):
            if y % 2 == 0: # even lines X-stabilizers
                i = y // 2
                for j in range(l):
                    x = 2 * j
                    anc = (x, y)
                    self.ancilla_coords_x.add(anc)
                    self.data_coords.add((x + 1, y))
                    self.stabilizers.append({'type': 'X', 'ancilla': anc, 'data': [
                        ( 2 * ((j + A[0][0]) % l), 2 * ((i+ A[0][1]) % m) + 1), 
                        ( 2 * ((j + A[1][0]) % l), 2 * ((i+ A[1][1]) % m) + 1),
                        ( 2 * ((j + A[2][0]) % l), 2 * ((i+ A[2][1]) % m) + 1),
                        ( 2 * ((j + B[0][0]) % l) + 1, 2 * ((i+ B[0][1]) % m)), 
                        ( 2 * ((j + B[1][0]) % l) + 1, 2 * ((i+ B[1][1]) % m)),
                        ( 2 * ((j + B[2][0]) % l) + 1, 2 * ((i+ B[2][1]) % m))]})
            elif y % 2 == 1: # odd lines Z-stabilizers
                i = y // 2
                for j in range(l):
                    x = 2 * j
                    anc = (x + 1, y)
                    self.ancilla_coords_z.add(anc)
                    self.data_coords.add((x, y))
                    self.stabilizers.append({'type': 'Z', 'ancilla': anc, 'data': [
                        ( 2 * ((j - B[0][0]) % l), 2 * ((i - B[0][1]) % m) + 1), 
                        ( 2 * ((j - B[1][0]) % l), 2 * ((i - B[1][1]) % m) + 1),
                        ( 2 * ((j - B[2][0]) % l), 2 * ((i - B[2][1]) % m) + 1),
                        ( 2 * ((j - A[0][0]) % l) + 1, 2 * ((i - A[0][1]) % m)), 
                        ( 2 * ((j - A[1][0]) % l) + 1, 2 * ((i - A[1][1]) % m)),
                        ( 2 * ((j - A[2][0]) % l) + 1, 2 * ((i - A[2][1]) % m))]})
            else:
                assert False #wrong input 

        self.qubit_coords = self.sort_coords(self.data_coords | self.ancilla_coords_x | self.ancilla_coords_z)
        self.data_coords = self.sort_coords(self.data_coords)
        self.ancilla_coords = self.sort_coords(self.ancilla_coords_x | self.ancilla_coords_z)
        self.ancilla_coords_x = self.sort_coords(self.ancilla_coords_x)
        self.ancilla_coords_z = self.sort_coords(self.ancilla_coords_z)
        
        self.coord_to_index = {coord: i for i, coord in enumerate(self.qubit_coords)}
        self.index_to_coord = {i: coord for coord, i in self.coord_to_index.items()}

        #Logical operators
        for i in range(k // 2):
            self.logical_X.append(multiply_monomial_by_polynomial(alpha[i], f, l, m, type='L'))
            self.logical_Z.append(multiply_monomial_by_polynomial(beta[i], transpose_exponents(h,l,m), l, m, type='L') 
                                  + multiply_monomial_by_polynomial(beta[i], transpose_exponents(g,l,m), l, m, type='R'))
            self.logical_Z.append(multiply_monomial_by_polynomial(beta[i], transpose_exponents(f,l,m), l, m, type='R'))
            self.logical_X.append(multiply_monomial_by_polynomial(alpha[i], g, l, m, type='L') 
                                  + multiply_monomial_by_polynomial(alpha[i], h, l, m, type='R'))
        
    
    def get_info(self):
        # The returned coordinates are all sorted
        return {
            'qubit_coords': self.qubit_coords,
            'data_coords': self.data_coords,
            'ancilla_coords': self.ancilla_coords,
            'ancilla_coords_x': self.ancilla_coords_x,
            'ancilla_coords_z': self.ancilla_coords_z,
            'stabilizers': self.stabilizers,
            'coord_to_index': self.coord_to_index,
            'index_to_coord': self.index_to_coord,
            'logical_Z': self.logical_Z,
            'logical_X': self.logical_X
        }
    
    def patch_shift(self, coords_diff: list[int]):
        pass

    def get_parity_check_matrix(self):
        pass
    
    def build_standard_sm_round(self, noise_profile: list, code_capacity = False) -> stim.Circuit:
        
        p1, p2, p_M, p_R = noise_profile
        n = self.n
        k = self.k
        d = self.d
        m = self.m
        l = self.l
        A = self.A
        B = self.B

        # This one-round circuit only contains operations and noise, without detectors
        circuit = stim.Circuit()

        # Before a sm round, insert data qubit errors
        data_indices = [self.coord_to_index[c] for c in self.data_coords]
        circuit.append_operation("DEPOLARIZE1", data_indices, p1)

        # TICK 1: Hadamard on X ancillas
        CNOT_pairs = []
        H_targets = [self.coord_to_index[coord] for coord in self.ancilla_coords_x]
        circuit.append_operation("H", H_targets)
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE1", H_targets, p1)
        for ancilla_z in self.ancilla_coords_z:
                anc = self.coord_to_index[ancilla_z]
                data_coord = ( 2* (( - A[0][0] + ancilla_z[0]//2) % l) +1 , 2* (( - A[0][1] + ancilla_z[1]//2) % m))
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [data, anc])
                    CNOT_pairs += [data, anc]
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE2", CNOT_pairs, p2)
        circuit.append("TICK")

        # TICK 2
        CNOT_pairs = []
        for ancilla_z in self.ancilla_coords_z:
                anc = self.coord_to_index[ancilla_z]
                data_coord = ( 2* (( - A[2][0] + ancilla_z[0]//2) % l) +1 , 2* (( - A[2][1] + ancilla_z[1]//2) % m))
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [data, anc])
                    CNOT_pairs += [data, anc]
        for ancilla_x in self.ancilla_coords_x:
                anc = self.coord_to_index[ancilla_x]
                data_coord = ( 2* (( A[1][0] + ancilla_x[0]//2) % l) , 2* (( A[1][1] + ancilla_x[1]//2) % m) + 1)
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [anc, data])
                    CNOT_pairs += [anc, data]
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE2", CNOT_pairs, p2)
        circuit.append("TICK")

        # TICK 3
        CNOT_pairs = []
        for ancilla_z in self.ancilla_coords_z:
                anc = self.coord_to_index[ancilla_z]
                data_coord = ( 2* (( - B[0][0] + ancilla_z[0]//2) % l) , 2* (( - B[0][1] + ancilla_z[1]//2) % m) + 1 )
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [data, anc])
                    CNOT_pairs += [data, anc]
        for ancilla_x in self.ancilla_coords_x:
                anc = self.coord_to_index[ancilla_x]
                data_coord = ( 2* (( B[1][0] + ancilla_x[0]//2) % l) + 1, 2* (( B[1][1] + ancilla_x[1]//2) % m))
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [anc, data])
                    CNOT_pairs += [anc, data]
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE2", CNOT_pairs, p2)
        circuit.append("TICK")

        # TICK 4
        CNOT_pairs = []
        for ancilla_z in self.ancilla_coords_z:
                anc = self.coord_to_index[ancilla_z]
                data_coord = ( 2* (( - B[1][0] + ancilla_z[0]//2) % l) , 2* (( - B[1][1] + ancilla_z[1]//2) % m) + 1 )
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [data, anc])
                    CNOT_pairs += [data, anc]
        for ancilla_x in self.ancilla_coords_x:
                anc = self.coord_to_index[ancilla_x]
                data_coord = ( 2* (( B[0][0] + ancilla_x[0]//2) % l) + 1, 2* (( B[0][1] + ancilla_x[1]//2) % m))
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [anc, data])
                    CNOT_pairs += [anc, data]
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE2", CNOT_pairs, p2)
        circuit.append("TICK")

        # TICK 5
        CNOT_pairs = []
        for ancilla_z in self.ancilla_coords_z:
                anc = self.coord_to_index[ancilla_z]
                data_coord = ( 2* (( - B[2][0] + ancilla_z[0]//2) % l) , 2* (( - B[2][1] + ancilla_z[1]//2) % m) + 1 )
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [data, anc])
                    CNOT_pairs += [data, anc]
        for ancilla_x in self.ancilla_coords_x:
                anc = self.coord_to_index[ancilla_x]
                data_coord = ( 2* (( B[2][0] + ancilla_x[0]//2) % l) + 1, 2* (( B[2][1] + ancilla_x[1]//2) % m))
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [anc, data])
                    CNOT_pairs += [anc, data]
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE2", CNOT_pairs, p2)
        circuit.append("TICK")

        # TICK 6
        CNOT_pairs = []
        for ancilla_z in self.ancilla_coords_z:
                anc = self.coord_to_index[ancilla_z]
                data_coord = ( 2* (( - A[1][0] + ancilla_z[0]//2) % l) +1 , 2* (( - A[1][1] + ancilla_z[1]//2) % m))
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [data, anc])
                    CNOT_pairs += [data, anc]
        for ancilla_x in self.ancilla_coords_x:
                anc = self.coord_to_index[ancilla_x]
                data_coord = ( 2* (( A[0][0] + ancilla_x[0]//2) % l) , 2* (( A[0][1] + ancilla_x[1]//2) % m) + 1)
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [anc, data])
                    CNOT_pairs += [anc, data]
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE2", CNOT_pairs, p2)
        circuit.append("TICK")

        # TICK 7
        CNOT_pairs = []
        for ancilla_x in self.ancilla_coords_x:
                anc = self.coord_to_index[ancilla_x]
                data_coord = ( 2* (( A[2][0] + ancilla_x[0]//2) % l) , 2* (( A[2][1] + ancilla_x[1]//2) % m) + 1)
                if data_coord in self.data_coords:
                    data = self.coord_to_index[data_coord]
                    circuit.append_operation("CNOT", [anc, data])
                    CNOT_pairs += [anc, data]
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE2", CNOT_pairs, p2)
        circuit.append("TICK")

        # TICK 8: Hadamard on X ancillas
        circuit.append_operation("H", H_targets)
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE1", H_targets, p1)
        circuit.append("TICK")

        # TICK 9: MR with SPAM noise
        measure_targets = [self.coord_to_index[coord] for coord in self.ancilla_coords]
        if not code_capacity:
            circuit.append_operation("X_ERROR", measure_targets, p_M)
        circuit.append_operation("MR", measure_targets)
        if not code_capacity:
            circuit.append_operation("X_ERROR", measure_targets, p_R)
        # Don't insert TICK here, insert TICK after specifying the detectors in the construction of the full circuit

        return circuit
    
    def build_full_BBcode_circuit(self, rounds: int, noise_profile: list, observable_type: str, code_capacity = False) -> stim.Circuit:
    
        p1, p2, p_M, p_R = noise_profile

        full_circuit = stim.Circuit()
        repeat_circuit = stim.Circuit()

        # QUBIT_COORDS annotations
        for coord, index in self.coord_to_index.items():
            full_circuit.append_operation("QUBIT_COORDS", [index], list(coord))

        # Initialization
        data_indices = [self.coord_to_index[c] for c in self.data_coords]
        ancilla_indices = [self.coord_to_index[c] for c in self.ancilla_coords]
        if observable_type == "Z":
            full_circuit.append_operation("R", data_indices)
            if not code_capacity:
                full_circuit.append_operation("X_ERROR", data_indices, p_R)
        else:
            full_circuit.append_operation("RX", data_indices)
            if not code_capacity:
                full_circuit.append_operation("Z_ERROR", data_indices, p_R)
        
        full_circuit.append_operation("R", ancilla_indices)
        if not code_capacity:
            full_circuit.append_operation("X_ERROR", ancilla_indices, p_R)
        full_circuit.append("TICK")

        # First round
        full_circuit += self.build_standard_sm_round(noise_profile)
        # the first round, detectors are the stabilizers already created by initializatio
        for k, ancilla in enumerate(reversed(self.ancilla_coords)):
            if observable_type == "Z":
                if ancilla in self.ancilla_coords_z:
                    full_circuit.append_operation("DETECTOR", [stim.target_rec(-k - 1)], list(ancilla) + [0])
            else:
                if ancilla in self.ancilla_coords_x:
                    full_circuit.append_operation("DETECTOR", [stim.target_rec(-k - 1)], list(ancilla) + [0])

        # Later rounds
        repeat_circuit.append("TICK")
        repeat_circuit += self.build_standard_sm_round(noise_profile, code_capacity=code_capacity)
        repeat_circuit.append_operation("SHIFT_COORDS", [], [0,0,1])
        # Insert detectors
        for k, ancilla in enumerate(reversed(self.ancilla_coords)):
            # ancilla_index = self.coord_to_index[ancilla]
            prev = -k - 1 - len(self.ancilla_coords)
            repeat_circuit.append_operation("DETECTOR", [stim.target_rec(-k - 1), stim.target_rec(prev)], list(ancilla) + [0])

        full_circuit += repeat_circuit * (rounds-1)

        # Final measurement of data qubits in Z or X basis
        if observable_type == "X":
            if not code_capacity:
                full_circuit.append_operation("Z_ERROR", data_indices, p_M)
            full_circuit.append_operation("MX", data_indices),
        else: # if it's "Z"
            if not code_capacity:
                full_circuit.append_operation("X_ERROR", data_indices, p_M)
            full_circuit.append_operation("M", data_indices)

        # Final detectors
        for stab in self.stabilizers:
            if stab['type'] == observable_type:
                anc = stab['ancilla']
                try:
                    ancilla_index = self.ancilla_coords[::-1].index(anc)
                    anc_rec = stim.target_rec(-ancilla_index - 1 - len(self.data_coords))
                    data_rec_targets = [stim.target_rec(-self.data_coords[::-1].index(q) - 1) for q in stab['data']]
                    full_circuit.append_operation("DETECTOR", [anc_rec] + data_rec_targets, list(anc) + [1])
                except ValueError:
                    continue
        
        # logical observables
        logical_key = 'logical_' + observable_type
        for p, logical_data in enumerate(getattr(self, logical_key)):
            data_indices = [self.data_coords[::-1].index(data) for data in logical_data]
            full_circuit.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(-k-1) for k in data_indices], p)

        return full_circuit
        
    @staticmethod
    def sort_coords(coords: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return sorted(coords, key=lambda c: (c[1], c[0]))
    
    @staticmethod
    def shift_coords(coords: List[Tuple[int, int]], dx: int, dy: int) -> List[Tuple[int, int]]:
        return [(x + dx, y + dy) for (x, y) in coords]
    
    @staticmethod
    def shift_coord(coord: Tuple[int, int], dx: int, dy: int) -> Tuple[int, int]:
        x, y = coord
        return (x + dx, y + dy)