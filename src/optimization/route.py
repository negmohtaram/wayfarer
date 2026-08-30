import math
from src.models.experience import Experience


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns distance in kilometers between two lat/lng points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def travel_time_minutes(exp1: Experience, exp2: Experience, mode: str = "walk") -> int:
    dist_km = haversine(exp1.location[0], exp1.location[1], exp2.location[0], exp2.location[1])
    speeds = {"walk": 4.5, "bike": 14.0, "transit": 25.0}
    speed = speeds.get(mode, 4.5)
    return max(5, int((dist_km / speed) * 60))


def nearest_neighbor_route(experiences: list[Experience], start_idx: int = 0) -> list[Experience]:
    """Greedy nearest-neighbor heuristic for TSP-like route ordering."""
    if len(experiences) <= 1:
        return experiences

    unvisited = list(experiences)
    route = [unvisited.pop(start_idx)]

    while unvisited:
        last = route[-1]
        nearest = min(
            unvisited,
            key=lambda e: haversine(last.location[0], last.location[1], e.location[0], e.location[1])
        )
        route.append(nearest)
        unvisited.remove(nearest)

    return route


def two_opt_improve(route: list[Experience], max_iterations: int = 50) -> list[Experience]:
    """2-opt improvement of a route."""
    if len(route) <= 3:
        return route

    def total_distance(r: list[Experience]) -> float:
        return sum(
            haversine(r[i].location[0], r[i].location[1], r[i + 1].location[0], r[i + 1].location[1])
            for i in range(len(r) - 1)
        )

    best = list(route)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if total_distance(candidate) < total_distance(best):
                    best = candidate
                    improved = True

    return best


def optimize_route(experiences: list[Experience]) -> list[Experience]:
    """Full route optimization: nearest neighbor + 2-opt improvement."""
    if not experiences:
        return []
    route = nearest_neighbor_route(experiences)
    return two_opt_improve(route)


def cluster_by_day(
    experiences: list[Experience],
    num_days: int,
    city_center: tuple[float, float],
) -> list[list[Experience]]:
    """
    Cluster experiences into geographically coherent day-groups.

    Each day greedily grows its own cluster from a seed experience,
    pulling in the nearest unassigned experience until the day quota is
    filled.  Seeds alternate between city-center-nearest (Day 1) and
    farthest-from-previous-cluster (later days) so consecutive days
    explore different parts of the city.

    Route optimisation within each cluster is left to the caller
    (optimizer_node runs optimize_route per cluster).
    """
    if not experiences:
        return [[] for _ in range(num_days)]

    per_day = max(1, len(experiences) // num_days)
    remaining = list(experiences)
    days: list[list[Experience]] = []

    for day_idx in range(num_days):
        if not remaining:
            days.append([])
            continue

        # Last day gets whatever is left
        quota = len(remaining) if day_idx == num_days - 1 else min(per_day, len(remaining))

        # Seed selection:
        # Day 0 → nearest to city centre (familiar entry point)
        # Other days → farthest from the previous cluster's centroid
        #              (push into a fresh part of the city)
        if day_idx == 0 or not days or not days[-1]:
            seed = min(
                remaining,
                key=lambda e: haversine(e.location[0], e.location[1], city_center[0], city_center[1]),
            )
        else:
            prev = days[-1]
            prev_lat = sum(e.location[0] for e in prev) / len(prev)
            prev_lng = sum(e.location[1] for e in prev) / len(prev)
            seed = max(
                remaining,
                key=lambda e: haversine(e.location[0], e.location[1], prev_lat, prev_lng),
            )

        cluster: list[Experience] = [seed]
        remaining.remove(seed)

        # Greedily absorb the nearest unassigned experience
        while len(cluster) < quota and remaining:
            anchor = cluster[-1]
            nearest = min(
                remaining,
                key=lambda e: haversine(e.location[0], e.location[1], anchor.location[0], anchor.location[1]),
            )
            cluster.append(nearest)
            remaining.remove(nearest)

        days.append(cluster)

    while len(days) < num_days:
        days.append([])

    return days[:num_days]
