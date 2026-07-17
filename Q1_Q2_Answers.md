# Question 1

## (a)

### i.

```python
def reliable_sightings(sightings):
    reliable = []
    for s in sightings:
        if s["reliable"] == True:
            reliable.append(s)

    reliable.sort(key=lambda s: s["time"])

    locations = []
    for s in reliable:
        locations.append(s["location"])
    return locations
```

### ii.

```python
def most_recent_sighting(sightings):
    latest = max(sightings, key=lambda s: s["time"])
    return (latest["spotter"], latest["location"])
```

### iii.

```python
def sighting_summary(sightings):
    summary = {}
    for s in sightings:
        loc = s["location"]
        if loc in summary:
            summary[loc] = summary[loc] + 1
        else:
            summary[loc] = 1
    return summary
```

## (b)

```python
graph = {
    "Hostel Gate":     {"CS Block": 4, "Library": 6},
    "CS Block":        {"Hostel Gate": 4, "Canteen": 3, "EE Block": 5},
    "Library":         {"Hostel Gate": 6, "Canteen": 2, "Football Ground": 7},
    "Canteen":         {"CS Block": 3, "Library": 2, "Football Ground": 4, "Admin Block": 6},
    "EE Block":        {"CS Block": 5, "Admin Block": 2},
    "Football Ground": {"Library": 7, "Canteen": 4, "Admin Block": 5},
    "Admin Block":     {"Canteen": 6, "EE Block": 2, "Football Ground": 5},
}
```

## (c)

### i.

```python
def dfs(graph, start, goal):
    stack = [[start]]
    visited = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbour in graph[node]:
                if neighbour not in visited:
                    stack.append(path + [neighbour])
    return None
```

### ii.

```python
import heapq

def ucs(graph, start, goal):
    pq = [(0, [start])]
    visited = set()
    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]
        if node == goal:
            return (path, cost)
        if node not in visited:
            visited.add(node)
            for neighbour in graph[node]:
                if neighbour not in visited:
                    new_cost = cost + graph[node][neighbour]
                    heapq.heappush(pq, (new_cost, path + [neighbour]))
    return None
```

```python
print(dfs(graph, "Hostel Gate", "Admin Block"))
print(ucs(graph, "Hostel Gate", "Admin Block"))
```

UCS is guaranteed to find Cutie by the cheapest route because it always expands the path with the lowest total cost first. DFS just goes as deep as it can and BFS only counts the number of edges, so both of them ignore the actual step costs and can return an expensive path.

# Question 2

## (a)

```python
import heapq

graph = {
    "Server":  {"Lounge": 2, "Floor1": 4},
    "Lounge":  {"Server": 2, "Floor2": 3},
    "Floor1":  {"Server": 4, "Floor2": 2, "Floor3": 6},
    "Floor2":  {"Lounge": 3, "Floor1": 2, "Room313": 4},
    "Floor3":  {"Floor1": 6, "Room313": 2},
    "Room313": {"Floor2": 4, "Floor3": 2},
}

h = {
    "Server": 7,
    "Lounge": 6,
    "Floor1": 5,
    "Floor2": 3,
    "Floor3": 2,
    "Room313": 0,
}

def a_star(graph, start, goal, heuristic):
    pq = [(heuristic[start], 0, [start])]
    visited = set()
    while pq:
        f, cost, path = heapq.heappop(pq)
        node = path[-1]
        if node == goal:
            return (path, cost)
        if node not in visited:
            visited.add(node)
            for neighbour in graph[node]:
                if neighbour not in visited:
                    g = cost + graph[node][neighbour]
                    f = g + heuristic[neighbour]
                    heapq.heappush(pq, (f, g, path + [neighbour]))
    return None

print(a_star(graph, "Server", "Room313", h))
```

Answer: path = Server -> Lounge -> Floor2 -> Room313, total cost = 9

## (b)

```python
coverage = [2, 4, 4, 7, 7, 7, 3, 9, 9, 5]
```

### i.

```python
def hill_climbing(coverage, start_position):
    current = start_position
    while True:
        best = current
        for n in [current - 1, current + 1]:
            if n >= 0 and n < len(coverage):
                if coverage[n] > coverage[best]:
                    best = n
        if best == current:
            return current
        current = best
```

### ii.

```python
print(hill_climbing(coverage, 1))
print(hill_climbing(coverage, 6))
```

From position 1 it gets stuck right there at position 1 (coverage 4). From position 6 it climbs to position 7 (coverage 9), which is the true best spot.

Position 1 fails because its neighbour at position 2 also has coverage 4, which is equal, not strictly better, so the algorithm stops on this plateau. It would have to pass through lower values to ever reach the peak at position 7/8, and hill climbing never goes downhill.

## (c)

```python
import random

def local_beam_search(coverage, k, num_iterations):
    states = random.sample(range(len(coverage)), k)
    for i in range(num_iterations):
        candidates = set(states)
        for s in states:
            for n in [s - 1, s + 1]:
                if n >= 0 and n < len(coverage):
                    candidates.add(n)
        states = sorted(candidates, key=lambda p: coverage[p], reverse=True)[:k]

    result = []
    for p in states:
        result.append((p, coverage[p]))
    return result

print(local_beam_search(coverage, 3, 10))
```

# Question 3

## (a)

```python
import math

data = [
    ("Fahad",    7, 1, 90, "Pass"),
    ("Farhan",   3, 4, 60, "Fail"),
    ("Ali",      6, 2, 85, "Pass"),
    ("Muneeb",   2, 5, 55, "Fail"),
    ("Neelam",   8, 0, 95, "Pass"),
    ("Haleema",  4, 3, 65, "Fail"),
    ("Shahmeer", 7, 2, 88, "Pass"),
    ("Farooq",   3, 4, 58, "Fail"),
]
```

### i.

```python
def euclidean_distance(p1, p2):
    total = 0
    for i in range(len(p1)):
        total = total + (p1[i] - p2[i]) ** 2
    return math.sqrt(total)
```

### ii.

```python
def knn_predict(data, query, k):
    distances = []
    for name, hours, chai, attendance, result in data:
        d = euclidean_distance((hours, chai, attendance), query)
        distances.append((d, name, result))

    distances.sort()
    nearest = distances[:k]

    pass_votes = 0
    fail_votes = 0
    for d, name, result in nearest:
        if result == "Pass":
            pass_votes = pass_votes + 1
        else:
            fail_votes = fail_votes + 1

    if pass_votes > fail_votes:
        return "Pass"
    elif fail_votes > pass_votes:
        return "Fail"
    else:
        return nearest[0][2]
```

### iii.

```python
query = (5, 3, 70)
print(knn_predict(data, query, 3))
print(knn_predict(data, query, 5))
```

With k = 3 the nearest students are Haleema, Farhan and Farooq, all Fail, so prediction is Fail. With k = 5, Ali (Pass) and Muneeb (Fail) also join, giving 4 Fail vs 1 Pass, so still Fail. Yes, they agree. I would trust k = 3 more here, because the 3 closest students are genuinely similar to "You", while k = 5 starts pulling in students that are much farther away and less similar, which just adds noise on such a small dataset.

## (b)

### i. PEAS for Robotaa

Performance measure: orders delivered to the correct location, delivery time / fewest steps taken, no collisions, battery usage.

Environment: the campus map from Question 1 (Hostel Gate, CS Block, Library, Canteen, EE Block, Football Ground, Admin Block), the paths between them, and students/obstacles moving around.

Actuators: wheels/motors for moving, brakes for stopping, the order compartment lid, and a display/speaker to notify the customer.

Sensors: camera, GPS/location sensor, bump/proximity sensors for obstacles, and a battery level sensor.

### ii. Task environment classification

Partially Observable: the robot's sensors only see its nearby surroundings, it cannot see the whole campus at once.

Stochastic: students and obstacles move unpredictably, so the same action does not always lead to the same result.

Sequential: each move affects the next one, the whole delivery route is built from a chain of decisions.

Dynamic: the campus keeps changing (people walking around) while the robot is still deciding what to do.

Continuous: the robot's actual position, speed and time change smoothly, not in fixed separate steps.

### iii. Agent type

Robotaa is best modelled as a Utility-Based agent. It does not just need to reach the delivery location (which a goal-based agent could do), it needs to pick the best route among many possible ones, i.e. the cheapest/fastest path with least battery use. Comparing how good different routes are is exactly what a utility function does.
