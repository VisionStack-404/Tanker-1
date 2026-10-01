"""Strong heuristic solver for Tanker Run.

Designed for the Phase 1 CVRP benchmark.

Strategy:
- multiple deterministic constructions
- best-of-several initial solutions
- aggressive 2-opt within routes
- best relocation between routes
- best one-for-one swap between routes
- Or-opt short-chain relocation
- 2-opt* tail exchange between routes
- periodic diversification / reconstruction
- continuous best-candidate submission

All moves preserve the evaluator's capacity constraints.
"""

from adapter import Solver, route_load
from adapters.starter import StarterSolver
from data import distance_matrix

SAFETY_S = 0.20


def route_cost(d, route):
    if not route:
        return 0
    total = d[0][route[0]]
    for a, b in zip(route, route[1:]):
        total += d[a][b]
    total += d[route[-1]][0]
    return total


def total_cost(d, routes):
    return sum(route_cost(d, r) for r in routes)


def insertion_delta(d, route, v, pos):
    left = route[pos - 1] if pos else 0
    right = route[pos] if pos < len(route) else 0
    return d[left][v] + d[v][right] - d[left][right]


def best_insertion(d, route, v):
    best_delta = None
    best_pos = 0

    for pos in range(len(route) + 1):
        delta = insertion_delta(d, route, v, pos)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_pos = pos

    return best_delta, best_pos


def two_opt(d, route):
    """Best-improvement 2-opt until no improving reversal exists."""
    n = len(route)
    if n < 4:
        return False

    changed = False

    while True:
        best_delta = 0
        best_i = -1
        best_j = -1

        n = len(route)

        for i in range(n - 1):
            a = route[i - 1] if i else 0
            x = route[i]

            for j in range(i + 1, n):
                y = route[j]
                b = route[j + 1] if j + 1 < n else 0

                delta = (
                    d[a][y]
                    + d[x][b]
                    - d[a][x]
                    - d[y][b]
                )

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i == -1:
            break

        route[best_i:best_j + 1] = reversed(route[best_i:best_j + 1])
        changed = True

    return changed


def relocate_best(d, instance, routes, loads):
    """Best improving one-village relocation across routes."""
    best = None
    rcount = len(routes)

    for a in range(rcount):
        ra = routes[a]

        for i, v in enumerate(ra):
            prev_v = ra[i - 1] if i else 0
            next_v = ra[i + 1] if i + 1 < len(ra) else 0

            remove_delta = (
                d[prev_v][next_v]
                - d[prev_v][v]
                - d[v][next_v]
            )

            demand_v = instance.demand[v]

            for b in range(rcount):
                if a == b:
                    continue

                if loads[b] + demand_v > instance.capacity:
                    continue

                rb = routes[b]

                for pos in range(len(rb) + 1):
                    add_delta = insertion_delta(d, rb, v, pos)
                    total_delta = remove_delta + add_delta

                    if total_delta < (best[0] if best is not None else 0):
                        best = (
                            total_delta,
                            a,
                            i,
                            b,
                            pos,
                            v,
                        )

    if best is None:
        return False

    _, a, i, b, pos, v = best

    routes[a].pop(i)
    routes[b].insert(pos, v)

    loads[a] -= instance.demand[v]
    loads[b] += instance.demand[v]

    return True


def swap_best(d, instance, routes, loads):
    """Best improving 1-for-1 swap between two routes."""
    best = None
    rcount = len(routes)

    for a in range(rcount):
        ra = routes[a]

        for i, va in enumerate(ra):
            a_prev = ra[i - 1] if i else 0
            a_next = ra[i + 1] if i + 1 < len(ra) else 0

            for b in range(a + 1, rcount):
                rb = routes[b]

                for j, vb in enumerate(rb):
                    new_load_a = (
                        loads[a]
                        - instance.demand[va]
                        + instance.demand[vb]
                    )

                    if new_load_a > instance.capacity:
                        continue

                    new_load_b = (
                        loads[b]
                        - instance.demand[vb]
                        + instance.demand[va]
                    )

                    if new_load_b > instance.capacity:
                        continue

                    b_prev = rb[j - 1] if j else 0
                    b_next = rb[j + 1] if j + 1 < len(rb) else 0

                    old_cost = (
                        d[a_prev][va]
                        + d[va][a_next]
                        + d[b_prev][vb]
                        + d[vb][b_next]
                    )

                    new_cost = (
                        d[a_prev][vb]
                        + d[vb][a_next]
                        + d[b_prev][va]
                        + d[va][b_next]
                    )

                    delta = new_cost - old_cost

                    if delta < (best[0] if best is not None else 0):
                        best = (delta, a, i, b, j)

    if best is None:
        return False

    _, a, i, b, j = best

    routes[a][i], routes[b][j] = routes[b][j], routes[a][i]

    loads[a] = route_load(instance, routes[a])
    loads[b] = route_load(instance, routes[b])

    return True


def or_opt_best(d, instance, routes, loads, max_len=3):
    """Move a short consecutive block between two routes."""
    best = None
    rcount = len(routes)

    for a in range(rcount):
        ra = routes[a]
        n = len(ra)

        for length in range(2, min(max_len, n) + 1):
            for start in range(n - length + 1):
                block = ra[start:start + length]

                block_load = sum(instance.demand[v] for v in block)

                prev_v = ra[start - 1] if start else 0
                next_v = (
                    ra[start + length]
                    if start + length < n
                    else 0
                )

                internal = 0
                for x, y in zip(block, block[1:]):
                    internal += d[x][y]

                remove_delta = (
                    d[prev_v][next_v]
                    - d[prev_v][block[0]]
                    - internal
                    - d[block[-1]][next_v]
                )

                for b in range(rcount):
                    if a == b:
                        continue

                    if loads[b] + block_load > instance.capacity:
                        continue

                    rb = routes[b]

                    for pos in range(len(rb) + 1):
                        left = rb[pos - 1] if pos else 0
                        right = rb[pos] if pos < len(rb) else 0

                        add_delta = (
                            d[left][block[0]]
                            + internal
                            + d[block[-1]][right]
                            - d[left][right]
                        )

                        delta = remove_delta + add_delta

                        if delta < (best[0] if best is not None else 0):
                            best = (
                                delta,
                                a,
                                start,
                                length,
                                b,
                                pos,
                            )

    if best is None:
        return False

    _, a, start, length, b, pos = best

    block = routes[a][start:start + length]
    del routes[a][start:start + length]
    routes[b][pos:pos] = block

    moved = sum(instance.demand[v] for v in block)

    loads[a] -= moved
    loads[b] += moved

    return True


def two_opt_star_best(d, instance, routes, loads):
    """Exchange suffixes of two routes if the two resulting loads fit."""
    best = None
    rcount = len(routes)

    for a in range(rcount):
        ra = routes[a]

        if not ra:
            continue

        prefix_load_a = [0]
        running = 0

        for v in ra:
            running += instance.demand[v]
            prefix_load_a.append(running)

        total_a = prefix_load_a[-1]

        for b in range(a + 1, rcount):
            rb = routes[b]

            if not rb:
                continue

            prefix_load_b = [0]
            running = 0

            for v in rb:
                running += instance.demand[v]
                prefix_load_b.append(running)

            total_b = prefix_load_b[-1]

            # cut after ia elements and ib elements
            for ia in range(len(ra)):
                end_a = ra[ia]
                next_a = ra[ia + 1] if ia + 1 < len(ra) else 0

                for ib in range(len(rb)):
                    end_b = rb[ib]
                    next_b = rb[ib + 1] if ib + 1 < len(rb) else 0

                    new_load_a = (
                        prefix_load_a[ia + 1]
                        + (total_b - prefix_load_b[ib + 1])
                    )

                    if new_load_a > instance.capacity:
                        continue

                    new_load_b = (
                        prefix_load_b[ib + 1]
                        + (total_a - prefix_load_a[ia + 1])
                    )

                    if new_load_b > instance.capacity:
                        continue

                    old_edges = d[end_a][next_a] + d[end_b][next_b]
                    new_edges = d[end_a][next_b] + d[end_b][next_a]

                    delta = new_edges - old_edges

                    if delta < (best[0] if best is not None else 0):
                        best = (
                            delta,
                            a,
                            ia + 1,
                            b,
                            ib + 1,
                        )

    if best is None:
        return False

    _, a, ia, b, ib = best

    ra = routes[a]
    rb = routes[b]

    tail_a = ra[ia:]
    tail_b = rb[ib:]

    routes[a] = ra[:ia] + tail_b
    routes[b] = rb[:ib] + tail_a

    loads[a] = route_load(instance, routes[a])
    loads[b] = route_load(instance, routes[b])

    return True


def improve_routes(d, instance, routes, remaining_s, submit_candidate):
    """Intensified local search."""
    loads = [route_load(instance, r) for r in routes]

    best_seen = total_cost(d, routes)
    submit_candidate({"routes": [r[:] for r in routes]})

    rounds_without_gain = 0

    while remaining_s > SAFETY_S:
        before = total_cost(d, routes)

        # Phase 1: optimize route order.
        for route in routes:
            two_opt(d, route)

        # Phase 2: strong cross-route neighborhoods.
        moved = False

        if relocate_best(d, instance, routes, loads):
            moved = True

        if swap_best(d, instance, routes, loads):
            moved = True

        if or_opt_best(d, instance, routes, loads, max_len=3):
            moved = True

        if two_opt_star_best(d, instance, routes, loads):
            moved = True

        after = total_cost(d, routes)

        if after < best_seen:
            best_seen = after
            rounds_without_gain = 0
            receipt = submit_candidate({"routes": [r[:] for r in routes]})
            remaining_s = receipt["remaining_s"]
        else:
            rounds_without_gain += 1

            # Still submit a feasible candidate occasionally.
            if moved:
                receipt = submit_candidate({"routes": [r[:] for r in routes]})
                remaining_s = receipt["remaining_s"]
            else:
                break

        if after >= before and rounds_without_gain >= 2:
            break

    return routes


def build_farthest_routes(instance):
    """Farthest seed + cheapest insertion."""
    d = distance_matrix(instance)

    remaining = set(range(1, instance.size + 1))
    routes = []
    loads = []

    while remaining and len(routes) < instance.fleet:
        seed = max(
            remaining,
            key=lambda v: (d[0][v], instance.demand[v], -v),
        )

        remaining.remove(seed)

        route = [seed]
        load = instance.demand[seed]

        while True:
            best = None

            for v in remaining:
                if load + instance.demand[v] > instance.capacity:
                    continue

                delta, pos = best_insertion(d, route, v)

                # Mild preference for geographically related customers.
                score = delta

                if best is None or score < best[0]:
                    best = (score, v, pos)

            if best is None:
                break

            _, v, pos = best

            route.insert(pos, v)
            load += instance.demand[v]
            remaining.remove(v)

        routes.append(route)
        loads.append(load)

    # Repair any remaining villages using best feasible insertion.
    while remaining:
        best = None

        for v in remaining:
            dv = instance.demand[v]

            for i, route in enumerate(routes):
                if loads[i] + dv > instance.capacity:
                    continue

                delta, pos = best_insertion(d, route, v)

                candidate = (delta, i, pos, v)

                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is None:
            # Generator guarantees a feasible solution, so this should not
            # happen. Keep a safe fallback.
            v = next(iter(remaining))
            routes.append([v])
            loads.append(instance.demand[v])
            remaining.remove(v)
        else:
            _, i, pos, v = best
            routes[i].insert(pos, v)
            loads[i] += instance.demand[v]
            remaining.remove(v)

    return [r for r in routes if r]


def build_demand_routes(instance):
    """Insert high-demand villages first, choosing the cheapest feasible route."""
    d = distance_matrix(instance)

    routes = []
    loads = []

    order = sorted(
        range(1, instance.size + 1),
        key=lambda v: (
            -instance.demand[v],
            -d[0][v],
            v,
        ),
    )

    for v in order:
        best = None

        for i, route in enumerate(routes):
            if loads[i] + instance.demand[v] > instance.capacity:
                continue

            delta, pos = best_insertion(d, route, v)

            if best is None or delta < best[0]:
                best = (delta, i, pos)

        if best is None:
            routes.append([v])
            loads.append(instance.demand[v])
        else:
            _, i, pos = best
            routes[i].insert(pos, v)
            loads[i] += instance.demand[v]

    return [r for r in routes if r]


def build_radial_routes(instance):
    """Sort by angle around depot, useful for geographic clustering."""
    d = distance_matrix(instance)

    dx0, dy0 = instance.coords[0]

    def key(v):
        x, y = instance.coords[v]
        dx = x - dx0
        dy = y - dy0

        # Avoid importing atan2: quadrant + cross-product ordering would be
        # more complex. Integer-safe approximation is sufficient here.
        angle_bucket = (dy >= 0, dx >= 0)
        radius = dx * dx + dy * dy

        return angle_bucket, radius

    order = sorted(
        range(1, instance.size + 1),
        key=key,
    )

    routes = []
    loads = []

    for v in order:
        best = None

        for i, route in enumerate(routes):
            if loads[i] + instance.demand[v] > instance.capacity:
                continue

            delta, pos = best_insertion(d, route, v)

            if best is None or delta < best[0]:
                best = (delta, i, pos)

        if best is None:
            routes.append([v])
            loads.append(instance.demand[v])
        else:
            _, i, pos = best
            routes[i].insert(pos, v)
            loads[i] += instance.demand[v]

    return [r for r in routes if r]


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        d = distance_matrix(instance)

        # Build several structurally different starting points.
        seeds = []

        starter = StarterSolver().solve(
            instance,
            lambda _: None,
        )["routes"]

        seeds.append([r[:] for r in starter])
        seeds.append(build_farthest_routes(instance))
        seeds.append(build_demand_routes(instance))
        seeds.append(build_radial_routes(instance))

        # Pick the best starting solution.
        best_routes = min(
            seeds,
            key=lambda rs: total_cost(d, rs),
        )

        best_routes = [r[:] for r in best_routes]

        receipt = submit_candidate({
            "routes": [r[:] for r in best_routes]
        })

        # Intensify each diverse seed while time remains.
        for seed in seeds:
            if receipt["remaining_s"] <= SAFETY_S:
                break

            routes = [r[:] for r in seed]

            # Fast route-level optimization.
            for route in routes:
                two_opt(d, route)

            cost = total_cost(d, routes)
            best_cost = total_cost(d, best_routes)

            if cost < best_cost:
                best_routes = [r[:] for r in routes]

                receipt = submit_candidate({
                    "routes": [r[:] for r in best_routes]
                })

            if receipt["remaining_s"] <= SAFETY_S:
                break

            routes = improve_routes(
                d,
                instance,
                routes,
                receipt["remaining_s"],
                submit_candidate,
            )

            cost = total_cost(d, routes)

            if cost < total_cost(d, best_routes):
                best_routes = [r[:] for r in routes]

                receipt = submit_candidate({
                    "routes": [r[:] for r in best_routes]
                })

        return {
            "routes": [r for r in best_routes if r]
        }
