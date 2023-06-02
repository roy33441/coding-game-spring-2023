from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Set, cast
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
    base_distance: int = 0
    routes: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    grade_neigbors: float = 0
    closest_base_distance: int = 0
    closest_enemy_base_distance: int = 0
    closest_ant_distance: int = 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def debug(message: Any):
    return "MESSAGE " + str(message)


def beacon(cell_index: int, strength: int = 1):
    return f"BEACON {cell_index} {strength}"


def line(src_cell: int, dst_cell: int, strength: int = 1):
    return f"LINE {src_cell} {dst_cell} {strength}"


def find_same_neighbors(first_cell: Cell, second_cell: Cell) -> List[int]:
    return [
        neighbor
        for neighbor in first_cell.neighbors
        if neighbor in second_cell.neighbors
    ]


def make_route_for_target(
    src_cell: Cell, dst_cell: Cell, cells: List[Cell]
) -> List[int]:
    beacons: List[int] = [src_cell.index]
    route = src_cell.routes[dst_cell.index][1]
    for cell_index in range(1, len(route) - 1):
        same_neighbors = find_same_neighbors(
            cells[route[cell_index - 1]], cells[route[cell_index + 1]]
        )
        best_cell = (
            route[cell_index]
            if len(same_neighbors) == 1
            else min(
                same_neighbors,
                key=lambda neighbor: grade_beacon(cells[neighbor]),
            )
        )
        beacons.append(best_cell)
    beacons.append(dst_cell.index)
    return beacons


def is_route_ready(route: Tuple[List[int], int, int], cells: List[Cell]) -> bool:
    beacons, route_strength, num_ants = route
    for beacon in beacons:
        if cells[beacon].my_ants < route_strength:
            return False
    return True


def make_lines(routes: List[Tuple[List[int], int, int]]) -> str:
    actions: List[str] = []
    for route in routes:
        beacons, route_strength, num_ants = route
        route_actions: List[str] = [beacon(beacons[0], route_strength)]
        num_ants -= route_strength
        over_ants = max(0, cells[beacons[0]].my_ants - route_strength)
        # if game_turn == 6:
        #     raise Exception(num_ants)
        # If passing from cell to antoher cell and it's passing the chain strength also the current chain
        # should over some ants
        # For Example:
        # Ants: 6 -> 4 -> 1 -> 2 And the chain strentgth is 3 the
        # Result should be: 3 -> 4 -> 3 -> 3
        for cell_index in beacons[1:-1]:
            current_cell: Cell = cells[cell_index]
            current_beacon_strength = max(0, route_strength - over_ants)
            if current_beacon_strength != 0:
                route_actions.append(beacon(cell_index, current_beacon_strength))
            num_ants -= current_beacon_strength
            over_ants = max(0, current_cell.my_ants - route_strength)
        route_actions.append(beacon(beacons[-1], num_ants))
        actions.append(";".join(route_actions))
    return ";".join(actions)


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
        [None for _ in cells] for _ in cells
    ]
    distances = [[len(cells) for _ in cells] for _ in cells]
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
            grade += 7 if is_ending_of_game(cells) else 2
        elif cells[neighbor].cell_type == CellType.EGG:
            grade += 8 if is_beggining_of_game(cells) else 3
    return grade / 15


def set_cells_grade_neigbors(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        cell.grade_neigbors = set_grade_neigbors(cell)
    return cells


def grade_beacon(dst_cell: Cell) -> float:
    grade: float = dst_cell.closest_ant_distance * 0.2
    if dst_cell.opp_ants > dst_cell.my_ants:
        grade += 0.5
    if dst_cell.cell_type == CellType.EGG:
        grade -= 5 if is_beggining_of_game(cells) else 3
    if dst_cell.cell_type == CellType.CRYSTAL:
        grade -= 5 if is_ending_of_game(cells) else 2
    return grade


def grade_cell(src_cell: Cell, dst_cell: Cell) -> float:
    grade: float = src_cell.routes[dst_cell.index][0]
    # grade -= dst_cell.grade_neigbors
    grade += dst_cell.closest_ant_distance * 0.3
    grade += dst_cell.closest_base_distance * 0.3
    grade -= dst_cell.closest_enemy_base_distance * 0.15
    if dst_cell.opp_ants * grade > dst_cell.resources and dst_cell.my_ants == 0:
        grade += 10
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


def opponent_attack_chain_streangth(cell: Cell, cells: List[Cell]) -> int:
    streangths = [
        cell.opp_ants,
        max([cells[n].opp_ants for n in cell.neighbors]),
    ]
    return min(streangths)


def make_chain(
    bases: List[int], chain_cells: List[Cell], cells: List[Cell], chain_length: int
) -> List[str]:
    actions: List[str] = []
    beacons: Set[int] = set()
    routes: List[Tuple[List[int], int, int]] = []
    num_ants_available = get_my_ants_amount(cells)
    unused_bases = bases.copy()
    for _ in range(chain_length):
        if not chain_cells:
            return actions
        options: List[Tuple[int, Cell, Cell]] = []
        for beacon in unused_bases if len(unused_bases) != 0 else beacons:
            best_resource = get_best_cell(cells[beacon], chain_cells)
            distance = cells[beacon].routes[best_resource.index][0]
            options.append((distance, cells[beacon], best_resource))
        distance, src, target = min(options, key=lambda option: option[0])
        if src.index in unused_bases:
            unused_bases.remove(src.index)

        chain_cells = [
            chain_cell
            for chain_cell in chain_cells
            if chain_cell.index not in beacons and chain_cell.index != target.index
        ]
        new_beacons = make_route_for_target(src_cell=src, dst_cell=target, cells=cells)
        # If already is a beacon that calculated so remove that from the calculation of the ants consumed
        if src.index in beacons:
            distance -= 1
            src = cells[new_beacons[1]]
            new_beacons = new_beacons[1:]

        ants_strength_for_target = max(
            ANTS_NEEDED_FOR_TARGET,
            opponent_attack_chain_streangth(target, cells) + 1,
        )
        sum_ants_for_target = (distance + 1) * ants_strength_for_target

        new_beacons_state = beacons.copy()
        new_beacons_state.update(src.routes[target.index][1])

        if num_ants_available - sum_ants_for_target < 0:
            continue

        num_ants_available -= sum_ants_for_target
        routes.append((new_beacons, ants_strength_for_target, sum_ants_for_target))
        beacons.update(new_beacons)

        # actions.append(new_actions)
        # actions.append(
        #     debug(
        #         f"Target from {src.index} To: {target.index} Total grade: {grade_cell(src, target)} D: {src.routes[target.index][0]} CB: {target.closest_base_distance} CEB: {target.closest_enemy_base_distance} GN: {target.grade_neigbors}"
        #     )
        # )
    if len(routes) == 0:
        pass
        # for base in bases:
        #     target = get_closest_cell(cells[base], get_crystal_cells(cells))
        #     ants_strength_for_target = max(
        #         ANTS_NEEDED_FOR_TARGET,
        #         opponent_attack_chain_streangth(target, cells) + 1,
        #     )
        #     distance = cells[base].routes[target.index][0]
        #     actions.append(
        #         make_lines(
        #             cells[base],
        #             target,
        #             cells,
        #             (distance + 1) * ants_strength_for_target,
        #         )[0]
        #     )
    else:
        used_ants = sum(route[2] for route in routes)
        actions.append(debug(f"{used_ants} ROUTES: {len(routes)}"))
        total_ants = get_my_ants_amount(cells)
        # Should never happend 🤷
        if total_ants < used_ants:
            raise Exception("wtf")
        else:
            added_index = 0
            # Find the first route that is missing ants and bonus him.
            for route in range(len(routes)):
                if not is_route_ready(routes[route], cells):
                    added_index = route
                    break
            # actions.append(debug(f"AI: {added_index} ORG: {routes[added_index][2]} NEW: {routes[added_index][2] + total_ants - used_ants}"))
            routes[added_index] = (
                routes[added_index][0],
                routes[added_index][1],
                routes[added_index][2] + total_ants - used_ants,
            )
        actions.extend([make_lines(routes)])

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
        crystal_cells = get_crystal_cells(cells)
        eggs_cells = get_eggs_cells(cells)
        set_cells_grade_neigbors(cells)
        set_cell_closest_ant_distance(cells)
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
            # debug(
            #     f"scored points: {left_points(cells)} total: {total_points(cells)} ratio: {left_points(cells)/total_points(cells)}"
            # ),
            # debug(f"turn: {game_turn} time: {time.time() - t} seconds"),
        ]

        do_actions(actions)
        game_turn += 1
