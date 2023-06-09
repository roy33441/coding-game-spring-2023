from __future__ import annotations
import math, time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple, Any, Dict, Set, cast
from enum import Enum
from collections import defaultdict
import sys


global game_turn, MAX_TURNS, MAX_DEEP
MAX_TURNS = 100
game_turn = 1
ANTS_NEEDED_FOR_TARGET_CRYSTAL = 3
ANTS_NEEDED_FOR_TARGET_EGG = 3


class CellType(Enum):
    EMPTY = 0
    EGG = 1
    CRYSTAL = 2


@dataclass
class Beacon:
    index: int
    strength: int

    def __str__(self) -> str:
        return f"BEACON {self.index} {self.strength}"


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
    closest_base: int = 0

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.index == other.index
        elif isinstance(other, int):
            return self.index == other
        return False


@dataclass
class BaseRouteInfo:
    beacons: List[int]
    strength: int
    route_ants: int
    origin: int
    is_primary: bool = True
    max_strength: int = 100000


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


def is_route_ready(route: BaseRouteInfo, cells: List[Cell]) -> bool:
    for beacon in route.beacons:
        if cells[beacon].my_ants < route.strength:
            return False
    return True


def find_route_line_direction(route: BaseRouteInfo, cells: List[Cell]) -> BaseRouteInfo:
    beacons = route.beacons
    amount = len(beacons) // 2
    first_beacons = [cells[beacon].my_ants for beacon in beacons[:amount]]
    second_beacons = [cells[beacon].my_ants for beacon in beacons[amount:]]
    if len(beacons) % 2 == 1:
        second_beacons = second_beacons[1:]
    if sum(second_beacons) > sum(first_beacons):
        route.beacons = route.beacons[::-1]
    return route


# If passing from cell to antoher cell and it's passing the chain strength also the current chain
# should over some ants
# For Example:
# Ants: 6 -> 4 -> 1 -> 2 And the chain strentgth is 3 the
# Result should be: 3 -> 4 -> 3 -> 3
def make_lines(routes: List[BaseRouteInfo], cells: List[Cell]) -> str:
    actions: List[str] = []
    for curr_route in routes.copy():
        route = replace(curr_route)
        # route = find_route_line_direction(route, cells)
        num_ants = route.route_ants
        route_strength = num_ants // len(route.beacons)

        if len(route.beacons) == 1:
            actions.append(beacon(route.beacons[0], num_ants))
            continue
        if is_route_ready(route, cells):
            actions.append(
                ";".join(
                    [
                        beacon(beacon_index, route_strength)
                        for beacon_index in route.beacons
                    ]
                )
            )
            continue

        current_beacons: List[Beacon] = [Beacon(route.beacons[0], route_strength)]
        num_ants -= route_strength
        start_beacon = 1
        over_ants = max(0, cells[route.beacons[0]].my_ants - route_strength)  # 14
        over_ants_history: List[int] = [0]
        route_ants: List[int] = [cells[beacon].my_ants for beacon in route.beacons]
        for loop_index, cell_index in enumerate(route.beacons[start_beacon:]):
            current_cell: Cell = cells[cell_index]
            prev_cell: Cell = cells[current_beacons[-1].index]
            needed = max(
                0,
                route_strength * (len(route.beacons) - loop_index - 2)
                - sum(route_ants[loop_index + 2 :]),
            )
            current_beacon_strength = max(
                0,
                route_strength - over_ants,
                min(current_cell.my_ants - needed, route_strength),
            )
            curr_beacon = Beacon(cell_index, current_beacon_strength)

            my_future_ants = current_cell.my_ants + over_ants  # 1
            prev_cell_ants = over_ants_history[-1] + prev_cell.my_ants  # 4
            if my_future_ants < route_strength:
                diff = prev_cell_ants - my_future_ants  # 2
                # if just 1 can't divide
                if diff > 1:
                    real_prev_cell_ants = prev_cell.my_ants  # 4
                    add_to_me = diff // 2  # 1
                    curr_beacon.strength += (
                        current_beacons[-1].strength
                        + add_to_me
                        - real_prev_cell_ants  # 3
                    )
                    current_beacons[-1].strength = real_prev_cell_ants - add_to_me  # 3

            current_beacons.append(curr_beacon)
            num_ants -= current_beacon_strength
            over_ants_history.append(over_ants)
            over_ants = max(0, current_cell.my_ants - current_beacon_strength)

        # current_beacons.append(Beacon(route.beacons[-1], route_strength))
        # num_ants -= route_strength
        # If is from the base make the base with the most stregnth that he can
        beacon_index = len(current_beacons) - 1
        while num_ants > 0 and beacon_index != -1:
            if cells[current_beacons[beacon_index].index].my_ants < route_strength:
                current_beacons[beacon_index].strength += num_ants
                num_ants = 0
            beacon_index -= 1
        if beacon_index == -1 and num_ants > 0:
            current_beacons[-1].strength += num_ants
        actions.append(
            ";".join([str(beacon) for beacon in current_beacons if beacon.strength > 0])
        )
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
        [
            cell.initial_resources
            for cell in cells
            if cell.cell_type == CellType.CRYSTAL
        ],
    )


def left_points(cells: List[Cell]) -> int:
    return sum([cell.resources for cell in cells if cell.cell_type == CellType.CRYSTAL])


def enough_crystals(crystal_cells: List[Cell]):
    return (
        my_score + sum([cell.resources for cell in crystal_cells])
    ) / total_score_possible >= 0.50


def is_beggining_of_game(cells: List[Cell]) -> bool:
    RATIO_RESOURCES = 0.30
    RATIO_TURNS = MAX_TURNS // 3
    t_points = total_points(cells)
    l_points = left_points(cells)
    scored_points = max(t_points - l_points, 1)
    enemy_wins = enemy_score / t_points >= 0.15
    return (
        game_turn < RATIO_TURNS
        and scored_points / t_points < RATIO_RESOURCES
        and not enemy_wins
    )


def is_ending_of_game(cells: List[Cell]) -> bool:
    RATIO_RESOURCES = 0.65
    RATIO_TURNS = MAX_TURNS - (MAX_TURNS // 5)
    t_points = total_points(cells)
    l_points = left_points(cells)
    scored_points = max(t_points - l_points, 1)
    enemy_wins = enemy_score / t_points >= 0.45
    return (
        game_turn > RATIO_TURNS
        or scored_points / t_points > RATIO_RESOURCES
        or enemy_wins
    )


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


def grade_target_distance_from_bases_and_source(source: Cell, target: Cell) -> float:
    bases_diff = target.closest_base_distance / target.closest_enemy_base_distance
    distance: float = source.routes[target.index][0]
    CLOSE_BASE_WEIGHT = 0.5
    if target.cell_type == CellType.CRYSTAL and bases_diff < 0.4 and not ending:
        return 1000
    if target.cell_type == CellType.CRYSTAL and bases_diff <= 1.4:
        return (
            distance * 0.2
            + abs(target.closest_base_distance - target.closest_enemy_base_distance)
            * 0.8
        )
    return distance + target.closest_base_distance * CLOSE_BASE_WEIGHT


def grade_cell(src_cell: Cell, dst_cell: Cell) -> float:
    # grade: float = src_cell.routes[dst_cell.index][0]
    grade = grade_target_distance_from_bases_and_source(src_cell, dst_cell)
    grade += dst_cell.closest_ant_distance * 0.3
    grade += src_cell.routes[dst_cell.closest_base][0] * 0
    grade -= dst_cell.grade_neigbors
    grade -= 0.0001 if src_cell.index in my_bases else 0

    # grade += dst_cell.closest_base_distance * 0.3
    # grade -= dst_cell.closest_enemy_base_dis tance * 0.15
    if dst_cell.opp_ants * grade > dst_cell.resources and dst_cell.my_ants == 0:
        grade += 10
    if dst_cell.cell_type == CellType.EGG:
        change: float = 0
        if dst_cell.closest_base_distance / dst_cell.closest_enemy_base_distance >= 1.5:
            change = (
                -(dst_cell.closest_base_distance / dst_cell.closest_enemy_base_distance)
                * 2
            )
        elif dst_cell.closest_base_distance / dst_cell.closest_enemy_base_distance >= 3:
            change = -10
        elif is_beggining_of_game(cells):
            change += 10
        else:
            change += 3
        grade -= change
    return grade


def get_best_cell(src_cell, target_cells: List[Cell]) -> Cell:
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


def find_route_to_primary(
    origin: int, routes: List[BaseRouteInfo]
) -> List[BaseRouteInfo]:
    try:
        last_route = cast(
            BaseRouteInfo, next(route for route in routes if origin in route.beacons)
        )
    except:
        raise Exception(routes, origin)
    routes_to_primay: List[BaseRouteInfo] = [last_route]
    while last_route.is_primary == False:
        current_route = cast(
            BaseRouteInfo,
            next(route for route in routes if last_route.origin in route.beacons),
        )
        routes_to_primay.append(current_route)
        last_route = current_route
    return routes_to_primay


def opponent_attack_chain_streangth(
    cell: Cell, cells: List[Cell], to_check: bool = False
) -> int:
    streangths = [
        cell.opp_ants,
        max([cells[n].opp_ants for n in cell.neighbors]),
    ]
    if (
        to_check
        and cell.closest_enemy_base_distance / cell.closest_base_distance <= 0.7
    ):
        return max(streangths) + 1
    return min(streangths)


def make_chain(
    bases: List[int],
    chain_cells: List[Cell],
    cells: List[Cell],
    chain_length: int,
    last_routes: List[BaseRouteInfo],
) -> Tuple[List[str], List[BaseRouteInfo]]:
    actions: List[str] = []
    beacons: Set[int] = set()
    routes: List[BaseRouteInfo] = []
    crystal_targets: List[Cell] = []

    no_routes_chain_cells = chain_cells.copy()

    total_ants = get_my_ants_amount(cells)
    num_ants_available = total_ants
    unused_bases = bases.copy()
    current_time = time.time()
    for route in last_routes:
        ants_strength_for_target = max(
            ANTS_NEEDED_FOR_TARGET_CRYSTAL
            if cells[route.beacons[-1]].cell_type == CellType.CRYSTAL
            else ANTS_NEEDED_FOR_TARGET_EGG,
            opponent_attack_chain_streangth(cells[route.beacons[-1]], cells),
            True,
        )
        if (
            route.is_primary
            and cells[route.beacons[-1]].resources > cells[route.beacons[-1]].my_ants
            and ants_strength_for_target <= route.strength
            and is_route_ready(route, cells)
        ):
            route.strength = ants_strength_for_target
            route.route_ants = len(route.beacons) * ants_strength_for_target
            num_ants_available -= route.route_ants
            beacons.update(route.beacons)
            unused_bases.remove(route.beacons[0])
            routes.append(route)
            chain_cells = [
                chain_cell
                for chain_cell in chain_cells
                if chain_cell.index not in beacons
                and chain_cell.index != route.beacons[-1]
            ]
            crystal_targets.extend(
                [cells[route.beacons[-1]]]
                if cells[route.beacons[-1]].cell_type == CellType.CRYSTAL
                else []
            )
    # print(f"Turn left: {0.1 - time.time() + t}", file=sys.stderr)
    # print(f"F Past routes took: {current_time - t}", file=sys.stderr)

    for _ in range(chain_length):
        # print(
        #     f"starting searching {[r.beacons for r in routes]} {[c.index for c in crystal_targets]}",
        #     file=sys.stderr,
        # )
        if not chain_cells:
            break
        options: List[Tuple[int, Cell, Cell, float]] = []
        current_time = time.time()
        for beacon in unused_bases if len(unused_bases) != 0 else beacons:
            best_resource = get_best_cell(cells[beacon], chain_cells)
            distance = cells[beacon].routes[best_resource.index][0]
            grade = grade_cell(cells[beacon], best_resource)
            options.append(
                (
                    distance,
                    cells[beacon],
                    best_resource,
                    grade,
                )
            )

        distance, src, target, grade = min(
            options, key=lambda option: grade_cell(option[1], option[2])
        )
        # print(f"besttt: {target.index=} {grade=} {src.index=}", file=sys.stderr)

        new_beacons = make_route_for_target(src_cell=src, dst_cell=target, cells=cells)
        ants_strength_for_target = max(
            ANTS_NEEDED_FOR_TARGET_CRYSTAL
            if target.cell_type == CellType.CRYSTAL
            else ANTS_NEEDED_FOR_TARGET_EGG,
            opponent_attack_chain_streangth(target, cells) + 1,
            True,
        )
        route_max_strength = max(
            -10000, max(cells[cell_index].resources for cell_index in new_beacons[1:])
        )

        ants_strength_for_target = min(ants_strength_for_target, route_max_strength)
        # print(f"Turn left: {0.1 - time.time() + t}", file=sys.stderr)
        # # print(f"Find target took: {current_time - t}", file=sys.stderr)
        # If already is a beacon that calculated so remove that from the calculation of the ants consumed

        origin_src = replace(src)

        is_route_primary = True
        added_ants = 0
        routes_to_primary: List[BaseRouteInfo] = []
        current_time = time.time()
        if src.index in beacons:
            routes_to_primary = find_route_to_primary(origin_src.index, routes)
            added_ants = sum(
                max((ants_strength_for_target - route.strength), 0) * len(route.beacons)
                for route in routes_to_primary
            )
        # print(f"Turn left: {0.1 - time.time() + t}", file=sys.stderr)
        # print(f"F Search past beacons took: {current_time - t}", file=sys.stderr)

        sum_ants_for_target = (len(new_beacons)) * ants_strength_for_target
        new_beacons_state = beacons.copy()
        new_beacons_state.update(src.routes[target.index][1])
        will_remove_beacon = (
            ants_strength_for_target if new_beacons[0] in beacons else 0
        )

        if (
            num_ants_available - sum_ants_for_target - added_ants + will_remove_beacon
            < 0
        ):
            # print(
            #     f"Not enogth ants!!!!!!!!!! {added_ants=} {num_ants_available=} {sum_ants_for_target=} {new_beacons=}",
            #     file=sys.stderr,
            # )
            continue

        # Remove the unused bases
        if src.index in unused_bases:
            unused_bases.remove(src.index)

        current_time = time.time()
        if src.index in beacons:
            should_remove_from_route = True
            if src.index in bases and any(
                route for route in routes if route.beacons[0] == src.index
            ):
                previus_route = next(
                    route for route in routes if route.beacons[0] == src.index
                )
                if previus_route.strength < ants_strength_for_target:
                    previus_route.beacons = previus_route.beacons[1:]
                    previus_route.route_ants -= previus_route.strength
                    previus_route.is_primary = False
                    should_remove_from_route = False
            if should_remove_from_route:
                distance -= 1
                sum_ants_for_target -= ants_strength_for_target
                is_route_primary = False
                src = cells[new_beacons[1]]
                new_beacons = new_beacons[1:]

        # print(f"Turn left: {0.1 - time.time() + t}", file=sys.stderr)
        # # print(f"Fix strengths took: {current_time - t}", file=sys.stderr)
        # There is change in past routes
        if added_ants > 0:
            for route in routes_to_primary:
                route.strength = max(route.strength, ants_strength_for_target)
                route.route_ants = route.strength * len(route.beacons)

        num_ants_available -= sum_ants_for_target + added_ants

        MAX_ROUTE_SIZE = 6
        routes_to_merge = [
            route
            for route in routes
            if route.beacons[-1] == origin_src.index
            and route.strength == ants_strength_for_target
            and target.routes[route.beacons[0]][0] + 1
            >= len(route.beacons) + len(new_beacons)
            and len(route.beacons) < MAX_ROUTE_SIZE
            # and len(new_beacons) == 1
        ]
        # If is the ending of other route
        if len(new_beacons) < MAX_ROUTE_SIZE and routes_to_merge:
            merging_route = routes_to_merge[0]
            merging_route.beacons.extend(new_beacons)
            merging_route.route_ants += sum_ants_for_target
        else:
            routes.append(
                BaseRouteInfo(
                    new_beacons,
                    ants_strength_for_target,
                    sum_ants_for_target,
                    origin_src.index,
                    is_route_primary,
                    route_max_strength,
                )
            )
        beacons.update(new_beacons)

        crystal_targets.extend(
            [
                cells[beacon]
                for beacon in new_beacons
                if cells[beacon].cell_type == CellType.CRYSTAL
            ]
        )
        if enough_crystals(crystal_targets):
            # print(
            #     "Dude we gonna win chill",
            #     [c.index for c in crystal_targets],
            #     [r.beacons for r in routes],
            #     file=sys.stderr,
            # )
            break

        chain_cells = [
            chain_cell
            for chain_cell in chain_cells
            if chain_cell.index not in beacons and chain_cell.index != target.index
        ]
        if time.time() - t > 0.08:
            # print(f"Break no timeee", file=sys.stderr)
            break

    # print("Dang that", file=sys.stderr)
    if len(routes) == 0:
        # print(f"NO ROUTESSSSSSSSSSSSS", file=sys.stderr)
        options = []
        for beacon in unused_bases if len(unused_bases) != 0 else beacons:
            best_resource = get_best_cell(cells[beacon], no_routes_chain_cells)
            distance = cells[beacon].routes[best_resource.index][0]
            options.append(
                (
                    distance,
                    cells[beacon],
                    best_resource,
                    grade_cell(cells[beacon], best_resource),
                )
            )

        distance, src, target, grade = min(
            options, key=lambda option: grade_cell(option[1], option[2])
        )
        routes = [
            BaseRouteInfo(src.routes[target.index][1], 1, total_ants, src.index, True)
        ]
        actions.append(make_lines(routes, cells))
    else:
        current_time = time.time()
        used_ants = sum(route.route_ants for route in routes)

        while total_ants - used_ants > 0:
            strength_routes = [
                route for route in routes if route.max_strength > route.strength
            ]
            if not strength_routes:
                # print("There is no strength routes ", file=sys.stderr)
                strength_routes = routes
            smallest_strength = min(
                strength_routes, key=lambda route: route.strength
            ).strength
            strength_routes = (
                strength_routes[::-1]
                if is_beggining_of_game(cells)
                else strength_routes
            )

            route = next(
                current_route
                for current_route in strength_routes
                if current_route.strength == smallest_strength
            )
            beacons_amount = len(route.beacons)

            if beacons_amount > total_ants - used_ants:
                break
            route.strength += 1
            route.route_ants += beacons_amount
            used_ants += beacons_amount

        # print(f"Turn left: {0.1 - time.time() + t}", file=sys.stderr)
        # print(f"F Improve strengths took: {current_time - t}", file=sys.stderr)
        # Should never happend 🤷
        current_time = time.time()
        if total_ants < used_ants:
            # raise Exception("wtf", total_ants, used_ants, routes)
            return ([], [])
        elif len(routes) > 0:
            # print(f"Putting the leftovers {total_ants - used_ants}", file=sys.stderr)
            leftover_routes = routes[::-1] if is_beggining_of_game(cells) else routes
            added_index = 0
            # Find the first route that is missing ants and bonus him.
            for route_index in range(len(leftover_routes)):
                if not is_route_ready(leftover_routes[route_index], cells):
                    added_index = route_index
                    break
            leftover_routes[added_index] = BaseRouteInfo(
                leftover_routes[added_index].beacons,
                leftover_routes[added_index].strength,
                leftover_routes[added_index].route_ants + total_ants - used_ants,
                leftover_routes[added_index].origin,
                leftover_routes[added_index].is_primary,
            )
            routes = leftover_routes
            used_ants = sum([route.route_ants for route in routes])
            # print(f"{added_index=} {total_ants=} {used_ants=}", file=sys.stderr)
        # print(f"Turn left: {0.1 - time.time() + t}", file=sys.stderr)
        # print(f"F Put leftovers took: {current_time - t}", file=sys.stderr)
        # print(*routes, sep="\n", file=sys.stderr)
        # print(
        #     f"We tring use: {sum(route.route_ants for route in routes)} ants",
        #     file=sys.stderr,
        # )
        current_time = time.time()
        actions.extend([make_lines(routes.copy(), cells)])
        # print(f"Turn left: {0.1 - time.time() + t}", file=sys.stderr)
        # print(f"F Make lines took: {current_time - t}", file=sys.stderr)
    return [*actions], routes


def number_ants(cells: List[Cell]) -> int:
    return sum([cell.my_ants for cell in cells])


def update_cells_closest_base_distance(
    cells: List[Cell], bases: List[int]
) -> List[Cell]:
    for cell in cells:
        cell.closest_base = get_closest_cell(cell, [cells[b] for b in bases]).index
        cell.closest_base_distance = cell.routes[cell.closest_base][0]
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


def get_bases_ants_amount(my_ants: List[Cell], my_bases: List[Cell]) -> Dict[int, int]:
    bases_ants_amount: Dict[int, int] = defaultdict(int)
    for ant in my_ants:
        closest_base = get_closest_cell(ant, my_bases)
        bases_ants_amount[closest_base.index] += ant.my_ants
    return bases_ants_amount


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
    last_routes: List[BaseRouteInfo] = []
    total_score_possible = total_points(cells)

    # game loop
    while True:
        try:
            my_score, enemy_score = [int(i) for i in input().split()]
            cells = update_cells(cells)
            if game_turn != 1:
                t = time.time()
            crystal_cells = get_crystal_cells(cells)
            ending = is_ending_of_game(cells)
            eggs_cells = get_eggs_cells(cells)
            set_cells_grade_neigbors(cells)
            set_cell_closest_ant_distance(cells)
            # if is_beggining_of_game(cells) and game_turn < 5 and eggs_cells:
            #     target_cells = [*eggs_cells]
            if is_ending_of_game(cells) or number_ants(cells) > sum(
                crystal.resources for crystal in crystal_cells
            ):
                target_cells = [*crystal_cells]
            else:
                target_cells = [*crystal_cells, *eggs_cells]

            my_ants = get_my_ant_cells(cells)
            # bases_ants_amount = get_bases_ants_amount(my_ants, [cells[b] for b in my_bases])
            actions, current_routes = make_chain(
                my_bases,
                target_cells.copy(),
                cells,
                min(number_ants(cells) // 5, len(target_cells)),
                last_routes,
            )
            last_routes = current_routes

            do_actions(actions)
            # print(f"Turn took: {time.time() - t}", file=sys.stderr)
            game_turn += 1
        except:
            print("WAIT")
