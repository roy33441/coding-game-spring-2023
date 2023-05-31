from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Set, Union, cast
from enum import Enum


global game_turn, MAX_TURNS, MAX_DEEP
MAX_TURNS = 100
game_turn = 1
ANTS_NEEDED_FOR_TARGET = 3


class CellType(Enum):
    EMPTY = 0
    EGG = 1
    CRYSTAL = 2


@dataclass
class Cell:
    index: int
    cell_type: CellType
    resources: int
    initial_resources: int
    neighbors: list[int]
    my_ants: int
    opp_ants: int
    base_distance: int = 0
    routes: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    grade_neigbors: float = 0
    closest_base_distance: int = 0
    closest_enemy_base_distance: int = 0
    closest_ant_distance: int = 0
    bonus_grade: float = 0

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cell) and self.index == other.index


def debug(message: Any) -> str:
    return "MESSAGE " + str(message)


def beacon(cell_index: int, strength: int = 1) -> str:
    return f"BEACON {cell_index} {strength}"

def beacon_action(cell_index: int, strength: int = 1) -> str:
    return f"BEACON {cell_index} {strength}"


def line(src_cell: int, dst_cell: int, strength: int = 1) -> str:
    return f"LINE {src_cell} {dst_cell} {strength}"


def make_lines2(src_cell: Cell, dst_cell: Cell, strength: int = 1) -> str:
    s = [
        *[beacon(cell, strength) for cell in src_cell.routes[dst_cell.index][1]],
    ]
    return ";".join(s)


def update_streangth_with_leftover(beacons_streangth: Dict[int, int], beacons: Set[int], leftover: int, streangth: int):
    if leftover != 0:
        relevant_for_leftover = [beacon for beacon in beacons if cells[beacon].my_ants <= streangth][:leftover]
        for i in range(len(relevant_for_leftover) - 1):
            beacons_streangth[relevant_for_leftover[i]] += 1
    
        beacons_streangth[relevant_for_leftover[-1]] += leftover - len(relevant_for_leftover) + 1


def make_lines(bases_beacons: Dict[int, Set[int]], cells: List[Cell]) -> List[str]:
    beacons_streangth: Dict[int, int] = {}
    actions = []
    actions.append(debug(f"Bases beacons are {bases_beacons}"))
    ants_without_beacons = set([])
    bases_streangth: Dict[int, int] = {}
    for base, base_beacons in bases_beacons.items():
        if len(base_beacons) == 1:
            ants_without_beacons.add(base)
            continue
        beacons_amount = len(base_beacons)
        ants_amount = sum([cells[b].my_ants for b in base_beacons])
        streangth = ants_amount // beacons_amount
        bases_streangth[base] = streangth
        leftover = ants_amount - streangth * beacons_amount
        actions.append(debug(f"{base=} {beacons_amount=} {ants_amount=} {streangth=} {leftover=}"))
        for beacon in base_beacons:
            beacons_streangth[beacon] = beacons_streangth.get(beacon, 0) + streangth
        
        update_streangth_with_leftover(beacons_streangth, base_beacons, leftover, streangth)

    actions.append(debug(f"Ants without beacons1 {ants_without_beacons}"))
    ants_without_beacons.update(ant.index for ant in get_my_ant_cells(cells) if not ant.index in beacons_streangth)
    used_beacons: List[Cell] = list(cells[b] for b in beacons_streangth.keys())
    actions.append(debug(f"Ants without beacons2 {ants_without_beacons}"))
    for ant in ants_without_beacons:
        leftover = cells[ant].my_ants
        closest_beacon = get_closest_cell(cells[ant], used_beacons)
        closest_beacon_base = [base for base in bases_beacons if closest_beacon.index in bases_beacons[base]][0]
        beacons = bases_beacons[closest_beacon_base]
        streangth = bases_streangth[closest_beacon_base]
        actions.append(debug(f"Here123 {leftover=} {streangth=} {ant=}"))
        update_streangth_with_leftover(beacons_streangth, beacons, leftover, streangth)
            

    actions.extend([beacon_action(beacon, streangth) for beacon, streangth in beacons_streangth.items()])
    return actions

def initilize_cells() -> List[Cell]:
    cells: List[Cell] = []

    number_of_cells = int(input())  # amount of hexagonal cells in this map
    for i in range(number_of_cells):
        inputs = [int(j) for j in input().split()]
        cell_type = inputs[0]  # 0 for empty, 1 for eggs, 2 for crystal
        initial_resources = inputs[
            1
        ]  # the initial amount of eggs/crystals on this cell
        neigh_0 = inputs[2]  # the index of the neighbouring cell for each direction
        neigh_1 = inputs[3]
        neigh_2 = inputs[4]
        neigh_3 = inputs[5]
        neigh_4 = inputs[6]
        neigh_5 = inputs[7]
        cell: Cell = Cell(
            index=i,
            cell_type=list(CellType)[cell_type],
            resources=initial_resources,
            initial_resources=initial_resources,
            neighbors=list(
                filter(
                    lambda id: id > -1,
                    [neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5],
                )
            ),
            my_ants=0,
            opp_ants=0,
        )
        cells.append(cell)

    return cells


def initlize_bases() -> Tuple[List[int], List[int]]:
    number_of_bases = int(input())
    my_bases: list[int] = []
    for i in input().split():
        my_base_index = int(i)
        my_bases.append(my_base_index)
    opp_bases: list[int] = []
    for i in input().split():
        opp_base_index = int(i)
        opp_bases.append(opp_base_index)
    return my_bases, opp_bases


def update_cells(cells: List[Cell]) -> List[Cell]:
    for i in range(len(cells)):
        inputs = [int(j) for j in input().split()]
        resources = inputs[0]  # the current amount of eggs/crystals on this cell
        my_ants = inputs[1]  # the amount of your ants on this cell
        opp_ants = inputs[2]  # the amount of opponent ants on this cell

        cells[i].resources = resources
        cells[i].my_ants = my_ants
        cells[i].opp_ants = opp_ants
        if cells[i].resources == 0:
            cells[i].cell_type = CellType.EMPTY

    return cells


def set_cell_closest_ant_distance(cells: List[Cell]) -> List[Cell]:
    my_ants_cells = get_my_ant_cells(cells)
    for cell in cells:
        closest_ant = get_closest_cell(cell, my_ants_cells)
        cell.closest_ant_distance = cell.routes[closest_ant.index][0]

    return cells


def do_actions(actions: List[str]) -> None:
    if len(actions) == 0:
        print("WAIT")
    else:
        print(";".join(actions))


def get_crystal_cells(cells: List[Cell]) -> List[Cell]:
    crystals = [
        cell
        for cell in cells
        if cell.cell_type == CellType.CRYSTAL
        and (
            cell.resources > cell.opp_ants
            or cell.my_ants >= opponent_attack_chain_streangth(cell, cells)
        )
    ]
    return (
        crystals
        if len(crystals) > 0
        else [cell for cell in cells if cell.cell_type == CellType.CRYSTAL]
    )


def get_eggs_cells(cells: List[Cell]) -> List[Cell]:
    return [
        cell
        for cell in cells
        if cell.cell_type == CellType.EGG
        and (
            cell.resources > cell.opp_ants
            or cell.my_ants >= opponent_attack_chain_streangth(cell, cells)
        )
    ]


def calculate_all_distances(cells: List[Cell]) -> List[Cell]:
    last_cell_in_route: List[List[Optional[Cell]]] = [
        [None for cell in cells] for cell in cells
    ]
    distances = [[len(cells) for cell in cells] for cell in cells]
    for cell in cells:
        # distance, cur_cell, last_cell
        queue: List[Tuple[int, Cell, Cell]] = [(0, cell, cell)]
        queue_index = 0
        while queue_index < len(queue):
            distance, current_cell, last_cell = queue[queue_index]
            queue_index += 1
            if distance >= distances[cell.index][current_cell.index]:
                continue
            distances[cell.index][current_cell.index] = distance
            last_cell_in_route[cell.index][current_cell.index] = last_cell
            for neighbor in current_cell.neighbors:
                if neighbor == -1:
                    continue
                if distances[cell.index][neighbor] > distance + 1:
                    queue.append((distance + 1, cells[neighbor], current_cell))

    for cell in cells:
        for other in cells:
            # if other.index != cell.index:
            cell.routes[other.index] = (
                distances[cell.index][other.index],
                get_route(cell, other, cast(List[List[Cell]], last_cell_in_route)),
            )

    return cells


def remove_cells_by_indexes(
    origin_cells: List[Cell], remove_indexes: Union[List[int], Set[int]]
) -> List[Cell]:
    return [cell for cell in origin_cells if cell.index not in remove_indexes]


def get_route(
    start_cell: Cell, end_cell: Cell, last_cell_in_route: List[List[Cell]]
) -> List[int]:
    route: List[int] = [end_cell.index]
    while end_cell != start_cell:
        end_cell = last_cell_in_route[start_cell.index][end_cell.index]
        route.append(end_cell.index)
    return route[::-1]


def total_points(cells: List[Cell]) -> int:
    return sum(
        [cell.initial_resources for cell in cells if cell.cell_type != CellType.EGG],
    )


def left_points(cells: List[Cell]) -> int:
    return sum([cell.resources for cell in cells if cell.cell_type == CellType.CRYSTAL])


def is_beggining_of_game(cells: List[Cell]) -> bool:
    RATIO_RESOURCES = 0.30
    RATIO_TURNS = MAX_TURNS // 3
    t_points = total_points(cells)
    l_points = left_points(cells)
    scored_points = max(t_points - l_points, 1)
    return game_turn < RATIO_TURNS and scored_points / t_points < RATIO_RESOURCES


def is_ending_of_game(cells: List[Cell]) -> bool:
    RATIO_RESOURCES = 0.65
    RATIO_TURNS = MAX_TURNS - (MAX_TURNS // 5)
    t_points = total_points(cells)
    l_points = left_points(cells)
    scored_points = max(t_points - l_points, 1)
    return game_turn > RATIO_TURNS or scored_points / t_points > RATIO_RESOURCES


def set_grade_neigbors(cell: Cell) -> float:
    grade = 0
    for neighbor in cell.neighbors:
        if neighbor == -1:
            continue
        if cells[neighbor].cell_type == CellType.EMPTY:
            grade += 1
        elif cells[neighbor].cell_type == CellType.CRYSTAL:
            grade += 10 if is_ending_of_game(cells) else 3
        elif cells[neighbor].cell_type == CellType.EGG:
            grade += 10 if is_beggining_of_game(cells) else 3
    return grade / 15


def set_cells_grade_neigbors(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        cell.grade_neigbors = set_grade_neigbors(cell)
    return cells


def grade_cell(src_cell: Cell, dst_cell: Cell) -> float:
    grade: float = src_cell.routes[dst_cell.index][0]
    # grade -= dst_cell.grade_neigbors * 0.5
    grade += dst_cell.closest_ant_distance * 0.8
    grade += dst_cell.closest_base_distance * 0.3
    grade -= dst_cell.closest_enemy_base_distance * 0.15
    grade += src_cell.bonus_grade
    if dst_cell.opp_ants * grade > dst_cell.resources and dst_cell.my_ants == 0:
        grade += 10
    return grade


def get_grade_bonus(cell: Cell, cells: List[Cell]) -> float:
    grade_bonus: float = 0
    if cell.cell_type == CellType.EGG:
        grade_bonus -= 10 if is_beggining_of_game(cells) else 3
    return grade_bonus


def set_cells_grade_bonus(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        cell.bonus_grade = get_grade_bonus(cell, cells)
    return cells


def get_best_cell(src_cell: Cell, target_cells: List[Cell]) -> Any:
    return min(
        target_cells,
        key=lambda cell: grade_cell(src_cell, cell),
    )


def get_closest_cell(src_cell: Cell, target_cells: List[Cell]) -> Cell:
    return min(
        target_cells,
        key=lambda dst_cell: src_cell.routes[dst_cell.index][0],
    )


def get_my_ant_cells(all_cells: List[Cell]) -> List[Cell]:
    return [cell for cell in all_cells if cell.my_ants > 0]


def get_my_ants_amount(cells: List[Cell]) -> int:
    return sum([cell.my_ants for cell in cells])


def calculate_strength(cell: Cell) -> int:
    distance_from_base = min([cell.routes[base][0] for base in my_bases])
    distance_from_enemy = min([cell.routes[base][0] for base in enemy_bases])
    if cell.opp_ants >= cell.my_ants:
        return 1 if distance_from_base < distance_from_enemy else 1
    return 1


def get_ants_beacons(cells: List[Cell], bases: List[int]) -> Set[int]:
    beacons: Set[int] = set()
    for cell in cells:
        for base in bases:
            if cell.index != base and all(
                cells[chain_cell].my_ants > 0 for chain_cell in cell.routes[base][1]
            ):
                beacons.update(cell.routes[base][1])
    return beacons


def get_best_option(
    src_cells: Set[int], cells: List[Cell], targets: List[Cell]
) -> Tuple[int, Cell, Cell]:
    options: List[Tuple[int, Cell, Cell]] = []
    for src_cell in src_cells:
        best_resource = get_best_cell(cells[src_cell], targets)
        distance = cells[src_cell].routes[best_resource.index][0]
        options.append((distance, cells[src_cell], best_resource))

    # best_option = min(options, key=lambda option: option[0])
    return min(options, key=lambda option: grade_cell(option[1], option[2]))


def opponent_attack_chain_streangth(cell: Cell, cells: List[Cell]) -> int:
    streangths = [
        cell.opp_ants,
        max([cells[n].opp_ants for n in cell.neighbors]),
    ]
    return min(streangths)  # TODO: to max.


def make_chain(
    bases: List[int],
    chain_cells: List[Cell],
    cells: List[Cell],
    chain_length: int,
    my_chain_ants: List[Cell],
) -> List[str]:
    ## TODO: multiply beacons amount in largest assumption.
    actions: List[str] = []
    beacons: Set[int] = set(bases)
    bases_beacons: Dict[int, Set[int]] = {base: set([base]) for base in bases}
    my_ant_indexes = [ant.index for ant in my_chain_ants]
    #srcs = beacons | set(my_ant_indexes)
    #srcs = beacons
    num_ants_available = get_my_ants_amount(cells)
    total_ants = get_my_ants_amount(cells)
    unused_bases: Set[int] = set(bases)

    for _ in range(chain_length):
        if not chain_cells:
            return actions

        distance, src, target = get_best_option(unused_bases if unused_bases else beacons, cells, chain_cells)
        if src.index in unused_bases:
            unused_bases.remove(src.index)

        ants_strength_for_target = max(
            ANTS_NEEDED_FOR_TARGET, opponent_attack_chain_streangth(target, cells) + 1
        )

        if src.index in beacons:
            closest_beacon = src
        else:
            raise AssertionError
            closest_beacon = get_closest_cell(
                src, [cells[beacon] for beacon in beacons]
            )
            distance += target.routes[closest_beacon.index][0]

        ants_needed_for_target = (distance + 1) * ants_strength_for_target

        chain_cells = [
            chain_cell
            for chain_cell in chain_cells
            if chain_cell.index not in beacons and chain_cell.index != target.index
        ]
        chain_cells = remove_cells_by_indexes(chain_cells, [*beacons, target.index])

        route_to_target = src.routes[target.index][1]
        #route_to_beacon = (
        #    src.routes[closest_beacon.index][1] if closest_beacon != src else []
        #)

        new_beacons_state = beacons.copy()
        new_beacons_state.update(route_to_target)
        ants_per_beacon = total_ants // len(new_beacons_state)
        if ants_per_beacon < ants_strength_for_target:
            continue

        num_ants_available -= ants_needed_for_target

        if num_ants_available < 0:
            break

        # if closest_beacon != src:
        #     actions.append(make_lines(src, closest_beacon, calculate_strength(target)))


        current_bases = [b for b in bases if src.index in bases_beacons[b]]
        if not current_bases:
            actions.append(debug(f"Src {src}. bases {bases}. bases_beacons : {bases_beacons}"))
        assert len(current_bases) > 0
        actions.append(debug(f"Current bases are {current_bases}"))
        for b in current_bases:
            bases_beacons[b].update(route_to_target)

        actions.append(debug(f"Bases beacons are {bases_beacons}"))
        
        #actions.append(make_lines(src, target, calculate_strength(target)))
        
        # actions.append(
        #     debug(
        #         f"Target from {src.index} To: {target.index} Total grade: {grade_cell(src, target)} D: {src.routes[target.index][0]} B: {target.bonus_grade} CB: {target.closest_base_distance} CEB: {target.closest_enemy_base_distance} GN: {target.grade_neigbors}"
        #     )
        # )
        beacons.update(route_to_target)
        
        #srcs = beacons | set(my_ant_indexes)
    actions.extend(make_lines(bases_beacons, cells))

    if len(actions) == 0:
        actions.extend(
            line(
                base,
                get_closest_cell(cells[base], get_crystal_cells(cells)).index,
            )
            for base in bases
        )
    return [*actions]


def number_ants(cells: List[Cell]) -> int:
    return sum([cell.my_ants for cell in cells])


def update_cells_closest_base_distance(
    cells: List[Cell], bases: List[int]
) -> List[Cell]:
    for cell in cells:
        closest_base = get_closest_cell(cell, [cells[b] for b in bases])
        cell.closest_base_distance = cell.routes[closest_base.index][0]
    return cells


def update_cells_closest_enemy_base_distance(
    cells: List[Cell], enemy_bases: List[int]
) -> List[Cell]:
    for cell in cells:
        closest_base = get_closest_cell(cell, [cells[b] for b in enemy_bases])
        cell.closest_enemy_base_distance = cell.routes[closest_base.index][0]
    return cells


def cell_closest_ant_distance(cell: Cell, my_ants: List[Cell]) -> int:
    closest_ant = get_closest_cell(cell, my_ants)
    return cell.routes[closest_ant.index][0]


def has_my_ants_chain_to_base(cell: Cell, base: int, my_ant_indexes: List[int]) -> bool:
    return all(cell in my_ant_indexes for cell in cell.routes[base][1])


def get_my_chain_ants(cells: List[Cell], my_bases: List[int]) -> List[Cell]:
    chain_ants = []
    my_ants = get_my_ant_cells(cells)
    my_ant_indexes = [ant.index for ant in my_ants]
    for ant in my_ants:
        if any(
            has_my_ants_chain_to_base(ant, base, my_ant_indexes) for base in my_bases
        ):
            chain_ants.append(ant)
    return chain_ants


if __name__ == "__main__":
    t = time.time()
    cells = initilize_cells()
    my_bases, enemy_bases = initlize_bases()
    base: Cell = cells[my_bases[0]]
    crystal_cells = get_crystal_cells(cells)
    eggs_cells = get_eggs_cells(cells)
    target_cells = [*crystal_cells, *eggs_cells]
    cells = calculate_all_distances(cells)
    cells = update_cells_closest_base_distance(cells, my_bases)
    cells = update_cells_closest_enemy_base_distance(cells, enemy_bases)

    # game loop
    while True:
        my_score, enemy_score = [int(i) for i in input().split()]
        cells = update_cells(cells)
        if game_turn != 1:
            t = time.time()
        my_chain_ants = get_my_chain_ants(cells, my_bases)
        crystal_cells = get_crystal_cells(cells)
        eggs_cells = get_eggs_cells(cells)
        set_cells_grade_neigbors(cells)
        set_cell_closest_ant_distance(cells)
        set_cells_grade_bonus(cells)
        if is_ending_of_game(cells):
            target_cells = [*crystal_cells]
        else:
            target_cells = [*crystal_cells, *eggs_cells]

        actions = [
            *make_chain(
                my_bases,
                target_cells.copy(),
                cells,
                min(number_ants(cells) // 5, len(target_cells)),
                my_chain_ants,
            ),
            debug(
                f"scored points: {left_points(cells)} total: {total_points(cells)} ratio: {left_points(cells)/total_points(cells)}"
            ),
            # debug(f"turn: {game_turn} time: {time.time() - t} seconds"),
        ]

        do_actions(actions)
        game_turn += 1
