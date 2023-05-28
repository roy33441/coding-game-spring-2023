from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from enum import Enum


global game_turn, MAX_TURNS, MAX_DEEP
MAX_TURNS = 100
game_turn = 1
ANTS_NEEDED_FOR_TARGET = 2


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
    routes: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    grade_neigbors: int = 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def debug(message: Any):
    return "MESSAGE " + str(message)


def beacon(cell_index: int, strength: int = 1):
    return f"BEACON {cell_index} {strength}"


def line(src_cell: int, dst_cell: int, strength: int = 1):
    return f"LINE {src_cell} {dst_cell} {strength}"


def make_lines(src_cell: Cell, dst_cell: Cell, strength: int = 1) -> str:
    s = [
        *[beacon(cell, strength) for cell in src_cell.routes[dst_cell.index][1]],
    ]
    return ";".join(s)


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
        and (cell.resources > cell.opp_ants or cell.my_ants >= cell.opp_ants)
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
        and (cell.resources > cell.opp_ants or cell.my_ants >= cell.opp_ants)
    ]


def calculate_all_distances(cells: List[Cell]) -> List[Cell]:
    last_cell_in_route = [[None for cell in cells] for cell in cells]
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
                get_route(cell, other, last_cell_in_route),
            )

    return cells


def get_route(
    start_cell: Cell, end_cell: Cell, last_cell_in_route: List[List[Cell]]
) -> List[Cell]:
    route: List[Cell] = [end_cell.index]
    while end_cell != start_cell:
        end_cell: Cell = last_cell_in_route[start_cell.index][end_cell.index]
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
            grade += 5 if is_ending_of_game(cells) else 3
        elif cells[neighbor].cell_type == CellType.EGG:
            grade += 5 if is_beggining_of_game(cells) else 3
    return grade / 5


def set_cells_grade_neigbors(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        cell.grade_neighbors = set_grade_neigbors(cell)
    return cells


def grade_cell(src_cell: Cell, dst_cell: Cell) -> float:
    grade = src_cell.routes[dst_cell.index][0]
    grade -= dst_cell.grade_neigbors
    if dst_cell.opp_ants * grade > dst_cell.resources and dst_cell.my_ants == 0:
        grade += 15
    if dst_cell.cell_type == CellType.EGG:
        grade -= 10 if is_beggining_of_game(cells) else 3
    return grade


def get_best_cell(src_cell, target_cells: List[Cell]) -> Any:
    return min(
        target_cells,
        key=lambda cell: grade_cell(src_cell, cell),
    )


def get_closest_cell(src_cell, target_cells: List[Cell]) -> Cell:
    return min(
        target_cells,
        key=lambda dst_cell: src_cell.routes[dst_cell.index][0],
    )


def get_my_ant_cells(all_cells: List[Cell]) -> List[Cell]:
    return [cell for cell in all_cells if cell.my_ants >= 0]


def get_my_ants_amount(cells: List[Cell]) -> int:
    return sum([cell.my_ants for cell in cells])


def calculate_strength(cell: Cell) -> int:
    distance_from_base = min([cell.routes[base][0] for base in my_bases])
    distance_from_enemy = min([cell.routes[base][0] for base in enemy_bases])
    if cell.opp_ants >= cell.my_ants:
        return 1 if distance_from_base < distance_from_enemy else 1
    return 1


def get_ants_beacons(cells: List[Cell], bases: List[int]) -> List[int]:
    beacons: List[int] = set()
    for cell in cells:
        for base in bases:
            if cell.index != base and all(
                cells[chain_cell].my_ants > 0 for chain_cell in cell.routes[base][1]
            ):
                beacons.update(cell.routes[base][1])
    return beacons


def make_default_chains(
    bases: List[int], chain_cells: List[Cell], cells: List[Cell]
) -> Tuple[List[str], int, List[Cell]]:
    actions = []
    current_chain_cells = chain_cells.copy()
    used_ants = 0
    new_beacons: List[int] = []
    max_ants = number_ants(cells)
    for base in bases:
        closest_resource = get_best_cell(cells[base], current_chain_cells)
        distance = cells[base].routes[closest_resource.index][0]
        used_ants += max(ANTS_NEEDED_FOR_TARGET, closest_resource.opp_ants + 1) * (
            distance + 1
        )
        if used_ants > max_ants and len(actions) > 0:
            break
        actions.append(
            make_lines(
                cells[base], closest_resource, calculate_strength(closest_resource)
            )
        )
        actions.append(
            debug(
                f"used ants: {used_ants} going to: {closest_resource.index} from: {base} distance: {distance}"
            )
        )
        new_beacons.extend(cells[base].routes[closest_resource.index][1])

    return actions, used_ants, new_beacons


def make_chain(
    bases: List[int], chain_cells: List[Cell], cells: List[Cell], chain_length: int
) -> List[str]:
    actions, used_ants, used_targets = make_default_chains(bases, chain_cells, cells)
    beacons: List[int] = set(
        [
            *[base for base in bases],
            *used_targets,
        ]
    )
    num_ants_available = get_my_ants_amount(cells) - used_ants

    chain_cells = [c for c in chain_cells if c.index not in used_targets]
    for _ in range(chain_length):
        if not chain_cells:
            return actions
        options: List[Tuple[int, Cell, Cell]] = []
        for beacon in beacons:
            closest_resource = get_best_cell(cells[beacon], chain_cells)
            distance = cells[beacon].routes[closest_resource.index][0]
            options.append((distance, cells[beacon], closest_resource))
        best_option = min(options, key=lambda option: option[0])
        ants_needed_for_target = max(
            ANTS_NEEDED_FOR_TARGET, best_option[2].opp_ants + 1
        )
        num_ants_available -= (best_option[0] + 1) * ants_needed_for_target
        if num_ants_available < 0:
            break
        actions.append(
            make_lines(
                best_option[1], best_option[2], calculate_strength(best_option[2])
            )
        )
        beacons.update(best_option[1].routes[best_option[2].index][1])
        chain_cells = [
            c
            for c in chain_cells
            if c.index not in beacons and c.index != closest_resource.index
        ]
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


if __name__ == "__main__":
    t = time.time()
    cells = initilize_cells()
    my_bases, enemy_bases = initlize_bases()
    base: Cell = cells[my_bases[0]]
    crystal_cells = get_crystal_cells(cells)
    eggs_cells = get_eggs_cells(cells)
    target_cells = [*crystal_cells, *eggs_cells]
    cells = calculate_all_distances(cells)

    # game loop
    while True:
        cells = update_cells(cells)
        if game_turn != 1:
            t = time.time()
        crystal_cells = get_crystal_cells(cells)
        eggs_cells = get_eggs_cells(cells)
        set_cells_grade_neigbors(cells)
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
            ),
            debug(
                f"scored points: {left_points(cells)} total: {total_points(cells)} ratio: {left_points(cells)/total_points(cells)}"
            ),
            # debug(f"turn: {game_turn} time: {time.time() - t} seconds"),
        ]

        do_actions(actions)
        game_turn += 1
